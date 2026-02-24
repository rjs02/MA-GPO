import abc
import json
import time
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI


class OpenAIBatchAPIHelper(abc.ABC):
    def __init__(
        self,
        input_template: str,
        output_template: str,
        description: str | None = "No description provided",
        max_batch_size: int = 50_000,
        max_file_size_bytes: int = 100 * 1024 * 1024,
    ):
        self.input_template = input_template
        self.output_template = output_template
        self.description = description
        self.max_batch_size = max_batch_size
        self.max_file_size_bytes = max_file_size_bytes

        # Some basic sanity checks
        assert "{index}" in input_template, (
            "Input template must contain {index}. This will be replaced with the index of the batch file."
        )
        assert "{index}" in output_template, (
            "Output template must contain {index}. This will be replaced with the index of the batch file."
        )

    @abc.abstractmethod
    def create_request_iterator(
        self, existing_custom_ids: set[str] | None = None, handle_potential_issues: bool = False
    ) -> Iterator[dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_existing_custom_ids(self) -> set[str]:
        """
        Scans all existing batch output files and returns a set of all custom_ids found.
        """
        custom_ids = set()
        output_file_pattern = self.output_template.format(index="*")
        file_paths = sorted(Path(output_file_pattern).parent.glob(Path(output_file_pattern).name))

        for file_path in file_paths:
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        # The result files are JSONL files. Each line is a JSON object.
                        # The JSON object has a "custom_id" field.
                        data = json.loads(line)
                        if "custom_id" in data and data["custom_id"]:
                            custom_ids.add(data["custom_id"])
                    except (json.JSONDecodeError, KeyError):
                        # In rare cases, a line may not be a valid JSON or not contain the expected keys.
                        # Also, sometimes, the API returns an error for a request that does not have a custom_id
                        print(f"Warning: Could not process line in {file_path}: {line}")
        return custom_ids

    def write_data_to_batch_files(
        self, batch_file_content: list[dict[str, Any]], current_batch_file_id: int
    ):
        """
        Writes a list of requests to one or more batch files, ensuring no file exceeds MAX_FILE_SIZE_BYTES.
        """
        current_chunk_lines = []
        current_size = 0

        for request in batch_file_content:
            line = json.dumps(request) + "\n"
            line_bytes = line.encode("utf-8")

            if current_size + len(line_bytes) > self.max_file_size_bytes and current_chunk_lines:
                batch_file_name = self.input_template.format(index=current_batch_file_id)
                print(f"Writing batch file {batch_file_name}")

                # Create the directory if it doesn't exist
                Path(batch_file_name).parent.mkdir(parents=True, exist_ok=True)

                with open(batch_file_name, "w") as f:
                    f.writelines(current_chunk_lines)

                current_batch_file_id += 1
                current_chunk_lines = []
                current_size = 0

            current_chunk_lines.append(line)
            current_size += len(line_bytes)

        if current_chunk_lines:
            batch_file_name = self.input_template.format(index=current_batch_file_id)
            print(f"Writing batch file {batch_file_name}")
            with open(batch_file_name, "w") as f:
                f.writelines(current_chunk_lines)
            current_batch_file_id += 1

        return current_batch_file_id

    def write_requests_to_batch_files(self, requests_iterator: Iterator[dict[str, Any]]):
        """
        Takes an iterator of requests and writes them to batch files.
        """
        # Find the smallest index for a new batch file
        current_batch_file_id = 0
        while Path(self.input_template.format(index=current_batch_file_id)).exists():
            current_batch_file_id += 1

        batch_file_content = []
        requests_count = 0
        for request in requests_iterator:
            batch_file_content.append(request)
            requests_count += 1

            # If the batch file content is full, write it to a file
            if len(batch_file_content) >= self.max_batch_size:
                current_batch_file_id = self.write_data_to_batch_files(
                    batch_file_content, current_batch_file_id
                )
                batch_file_content = []

        # Write the remaining content to a file
        if batch_file_content:
            self.write_data_to_batch_files(batch_file_content, current_batch_file_id)

        if requests_count == 0:
            print("No new requests to prepare.")

    def prepare_batch_files(self):
        """
        This function prepares multiple batches of requests that can be submitted as OpenAI batch jobs.
        """
        requests_iterator = self.create_request_iterator()
        self.write_requests_to_batch_files(requests_iterator)

    def prepare_missing_batch_files(self):
        """
        Prepares batch files for requests that have not been processed yet.
        """
        print("Scanning existing results to find missing ones...")
        existing_custom_ids = self.get_existing_custom_ids()
        print(f"Found {len(existing_custom_ids)} existing results.")

        print("Preparing batch files for missing requests...")
        requests_iterator = self.create_request_iterator(
            existing_custom_ids=existing_custom_ids, handle_potential_issues=True
        )
        self.write_requests_to_batch_files(requests_iterator)

    def launch_batch_job(self, file_to_upload: str):
        client = OpenAI()

        print(f"Uploading batch file {file_to_upload}")
        batch_input_file = client.files.create(file=open(file_to_upload, "rb"), purpose="batch")

        print(f"Launching batch job for batch file {batch_input_file.id}")
        batch_input_file_id = batch_input_file.id

        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": self.description},
        )

        print(f"Batch job launched for batch file {batch_input_file_id}")

    def launch_batch_jobs(self):
        # Find all files in the parent directory that start with the prefix
        batch_file_pattern = self.input_template.format(index="*")
        print(f"Batch file pattern: {batch_file_pattern}")
        print(f"Raw pattern used: {Path(batch_file_pattern).name}")
        print(f"Parent directory: {Path(batch_file_pattern).parent}")
        file_paths = sorted(Path(batch_file_pattern).parent.glob(Path(batch_file_pattern).name))

        input_template_name = Path(self.input_template).name
        prefix, suffix = input_template_name.split("{index}")

        for file_path in file_paths:
            # Extract index from input file path
            index = file_path.name.removeprefix(prefix).removesuffix(suffix)

            # Construct output file path
            output_file_path = Path(self.output_template.format(index=index))

            # Check if output file exists
            if not output_file_path.exists():
                self.launch_batch_job(str(file_path))
            else:
                print(
                    f"Output file {output_file_path.name} already exists, skipping launch for {file_path.name}."
                )

    def retrieve_batch_results(self):
        client = OpenAI()

        # 1. Get recent batch jobs (at most 48 hours old)
        print("Retrieving recent batch jobs...")

        recent_batches = []
        # 48 hours ago as a unix timestamp
        forty_eight_hours_ago = time.time() - 48 * 60 * 60

        # The list operation on batches is auto-paginating.
        # It returns batches in descending order of creation time.
        for batch in client.batches.list(limit=100):
            if batch.created_at >= forty_eight_hours_ago:
                recent_batches.append(batch)
            else:
                # Batches are sorted by creation date, so we can stop once we are past the 48 hour window
                break

        if not recent_batches:
            print("No recent batch jobs found in the last 48 hours.")
            return

        print(f"Found {len(recent_batches)} batch jobs from the last 48 hours.")

        # 2. Match batch jobs to input files.
        # Get local batch input files
        batch_file_pattern = self.input_template.format(index="*")
        input_dir = Path(batch_file_pattern).parent
        local_input_files = {p.name for p in input_dir.glob(Path(batch_file_pattern).name)}

        if not local_input_files:
            print("No local batch input files found.")
            return

        print(f"Found local input files: {local_input_files}")

        # Create a mapping from OpenAI file ID to filename for batch files.
        print("Fetching file list from OpenAI to map file IDs to filenames...")
        openai_batch_files = client.files.list(purpose="batch")
        file_id_to_name = {f.id: f.filename for f in openai_batch_files}

        # Filter for relevant batches that are completed or cancelled
        relevant_batches = []
        for batch in recent_batches:
            input_filename = file_id_to_name.get(batch.input_file_id)
            if input_filename and input_filename in local_input_files:
                if batch.status in ["completed", "cancelled", "expired"]:
                    relevant_batches.append(batch)
                else:
                    print(
                        f"Batch {batch.id} (for {input_filename}) is not completed, cancelled or expired. Status: {batch.status}"
                    )

        if not relevant_batches:
            print("No completed, cancelled or expired batch jobs found for local input files.")
            return

        # 3. Download results
        print(f"Found {len(relevant_batches)} completed/cancelled/expired jobs to download results from.")
        for batch in relevant_batches:
            input_filename = file_id_to_name.get(batch.input_file_id)

            input_template_name = Path(self.input_template).name
            prefix, suffix = input_template_name.split("{index}")
            index = input_filename.removeprefix(prefix).removesuffix(suffix)
            output_file_path = Path(self.output_template.format(index=index))

            if output_file_path.exists():
                print(
                    f"Output file {output_file_path.name} already exists, skipping download for {input_filename}."
                )
                continue

            output_file_id = batch.output_file_id

            if not output_file_id:
                print(
                    f"Batch {batch.id} for {input_filename} is completed but has no output file ID."
                )
                continue

            print(f"Downloading results for batch {batch.id} (input file: {input_filename})...")

            try:
                file_response = client.files.content(output_file_id)

                with open(output_file_path, "wb") as f:
                    f.write(file_response.content)

                print(f"Results for {input_filename} saved to {output_file_path}")

            except Exception as e:
                print(f"Failed to download or save results for batch {batch.id}. Error: {e}")
