# Grouped Trainer for GPM with linear complexity optimization
#
# This trainer processes all responses for a prompt in a single forward pass,
# then computes pairwise losses using cached embeddings. This reduces forward
# passes from O(K²) to O(K) per prompt with K responses.

from abc import ABC
import os
import shutil
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm
import deepspeed

from general_preference.models import (
    GeneralPreferenceLoss,
    HighDimGeneralPreferenceLoss,
    HighDimGeneralPreferenceRegressionLoss,
    HighDimGeneralPreferenceMoELoss,
    HighDimGeneralPreferenceRegressionMoELoss,
)
from general_preference.models import SFTSumLoss


class GroupedPreferenceRewardTrainer(ABC):
    """
    Trainer for grouped preference data optimized for GPM's linear complexity.

    Key differences from standard trainer:
    1. Processes all K responses per prompt in ONE forward pass
    2. Caches embeddings φ(x, y) for each response
    3. Computes pairwise losses using indexed cached embeddings
    4. Supports conflicting preferences (same pair, different preference directions)

    Args:
        model: The GPM reward model
        strategy: Training strategy (handles distributed setup)
        optim: Optimizer
        train_dataloader: DataLoader with GroupedRewardDataset
        eval_dataloader: DataLoader with GroupedRewardDataset (optional)
        scheduler: Learning rate scheduler
        tokenizer: Tokenizer
        max_epochs: Number of training epochs
        tau: Temperature for preference loss
        value_head_dim: Dimension of GPM value head
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_epochs: int = 2,
        tau: float = 0.1,
        value_head_dim: int = 2,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.value_head_dim = value_head_dim

        # Initialize loss function for GPM
        if value_head_dim == 2 and not getattr(self.args, 'add_prompt_head', False):
            self.loss_fn = GeneralPreferenceLoss(tau)
            self.strategy.print("Using GeneralPreferenceLoss (dim=2)")
        else:
            assert value_head_dim % 2 == 0, "value_head_dim must be even for GPM!"
            if getattr(self.args, 'add_prompt_head', False):
                self.loss_fn = HighDimGeneralPreferenceMoELoss(
                    model=self.model,
                    value_head_dim=value_head_dim,
                    softmax_tau=tau
                )
                self.strategy.print(f"Using HighDimGeneralPreferenceMoELoss (dim={value_head_dim})")
            else:
                self.loss_fn = HighDimGeneralPreferenceLoss(tau, value_head_dim)
                self.strategy.print(f"Using HighDimGeneralPreferenceLoss (dim={value_head_dim})")

        self.ptx_loss_fn = SFTSumLoss(getattr(self.args, 'reward_scaler_beta', 0.1))

        self.margin_loss = getattr(self.args, 'margin_loss', False)
        self.compute_fp32_loss = getattr(self.args, 'compute_fp32_loss', False)

        # WandB setup
        self._wandb = None
        if getattr(self.args, 'use_wandb', None) and self.strategy.is_rank_0():
            import wandb
            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.args.use_wandb)
            wandb.init(
                entity=self.args.wandb_org,
                project=self.args.wandb_project,
                group=getattr(self.args, 'wandb_group', None),
                name=self.args.wandb_run_name,
                config=self.args.__dict__,
                reinit=True,
            )
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        """Main training loop for grouped data."""
        if args.eval_steps == -1:
            args.eval_steps = len(self.train_dataloader)
        if args.save_steps == -1:
            args.save_steps = float("inf")

        eval_loss_minimum = None
        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())

        for epoch in range(self.epochs):
            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            loss_mean = 0

            for batch in self.train_dataloader:
                # Grouped batch contains all responses and comparisons
                loss, prob, logs = self._grouped_training_step(batch)

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * loss.item()

                logs_dict = {
                    "preference_loss": loss.item(),
                    "prob": prob.item(),
                    "loss_mean": loss_mean,
                    **logs,
                }

                eval_loss_minimum = self.save_logs_and_checkpoints(
                    args, global_step, step_bar, eval_loss_minimum, logs_dict
                )
                torch.distributed.barrier()
                step_bar.update()
                global_step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    def _grouped_training_step(self, batch):
        """
        Single training step for grouped data.

        Key optimization: compute embeddings ONCE for all responses,
        then index into them for pairwise comparisons.
        """
        # Get batch data
        input_ids = batch["input_ids"].squeeze(1).to(torch.cuda.current_device())
        attention_mask = batch["attention_mask"].squeeze(1).to(torch.cuda.current_device())
        chosen_indices = batch["chosen_indices"].to(torch.cuda.current_device())
        rejected_indices = batch["rejected_indices"].to(torch.cuda.current_device())
        margins = batch["margins"].to(torch.cuda.current_device())

        num_responses = batch["num_responses"]
        num_comparisons = batch["num_comparisons"]

        # Single forward pass for ALL responses
        # This is the key efficiency gain: O(K) instead of O(K²)
        all_embeddings, _ = self.model.custom_forward(
            input_ids, attention_mask=attention_mask, return_output=False
        )

        if self.compute_fp32_loss:
            all_embeddings = all_embeddings.float()

        # Index into embeddings for each comparison
        chosen_embeddings = all_embeddings[chosen_indices]   # [num_comparisons, value_head_dim]
        rejected_embeddings = all_embeddings[rejected_indices]  # [num_comparisons, value_head_dim]

        # Compute loss
        if self.margin_loss:
            loss, prob = self.loss_fn(chosen_embeddings, rejected_embeddings, margins)
        else:
            loss, prob = self.loss_fn(chosen_embeddings, rejected_embeddings, None)

        logs = {
            "num_responses": num_responses,
            "num_comparisons": num_comparisons,
            "efficiency_ratio": num_comparisons / max(num_responses, 1),
        }

        return loss, prob, logs

    def evaluate(self, eval_dataloader, steps=0):
        """Evaluation loop for grouped data."""
        step_bar = tqdm(
            range(len(eval_dataloader)),
            desc=f"Eval stage of steps {steps}",
            disable=not self.strategy.is_rank_0(),
        )

        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            prob_sum = 0
            total_comparisons = 0

            for batch in eval_dataloader:
                input_ids = batch["input_ids"].squeeze(1).to(torch.cuda.current_device())
                attention_mask = batch["attention_mask"].squeeze(1).to(torch.cuda.current_device())
                chosen_indices = batch["chosen_indices"].to(torch.cuda.current_device())
                rejected_indices = batch["rejected_indices"].to(torch.cuda.current_device())
                margins = batch["margins"].to(torch.cuda.current_device())

                # Forward all responses
                all_embeddings, _ = self.model.custom_forward(
                    input_ids, attention_mask=attention_mask, return_output=False
                )

                # Index for comparisons
                chosen_embeddings = all_embeddings[chosen_indices]
                rejected_embeddings = all_embeddings[rejected_indices]

                # Compute loss
                if self.margin_loss:
                    loss, prob = self.loss_fn(chosen_embeddings, rejected_embeddings, margins)
                else:
                    loss, prob = self.loss_fn(chosen_embeddings, rejected_embeddings, None)

                loss_sum += loss.item()
                prob_sum += prob.item()
                total_comparisons += batch["num_comparisons"]
                step_bar.update()

            loss_mean = loss_sum / len(eval_dataloader)
            prob_mean = prob_sum / len(eval_dataloader)

            bar_dict = {
                "eval_loss_mean": loss_mean,
                "prob_mean": prob_mean,
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)

        self.model.train()
        torch.cuda.empty_cache()
        if self.strategy.is_rank_0():
            return loss_mean

    def save_logs_and_checkpoints(self, args, global_step, step_bar, eval_loss_minimum, logs_dict={}):
        """Save logs and checkpoints (same as base trainer)."""
        if global_step % args.logging_steps == 0:
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # Evaluation
        if global_step % args.eval_steps == 0 and self.eval_dataloader:
            try:
                eval_loss = self.evaluate(self.eval_dataloader, global_step)
                torch.distributed.barrier()
            except Exception as e:
                self.strategy.print(f"Error during evaluation: {str(e)}")
                raise

            if args.save_best_model:
                save_path = os.path.join(args.save_path, f"step_{global_step}")
                if eval_loss_minimum is None or eval_loss < eval_loss_minimum:
                    try:
                        self.strategy.save_model(self.model, self.tokenizer, save_path)
                        eval_loss_minimum = eval_loss
                    except Exception as e:
                        self.strategy.print(f"Error saving model: {str(e)}")
                        raise
                    try:
                        if self.strategy.is_rank_0():
                            self.clean_old_checkpoints(args.save_path, args.save_best_model)
                    except Exception as e:
                        self.strategy.print(f"Error deleting old checkpoint: {str(e)}")
                        raise

        # Regular checkpoint saving
        if global_step % args.save_steps == 0:
            tag = f"global_step_{global_step}"
            self.strategy.save_model(self.model, self.tokenizer, os.path.join(args.save_path, tag))

        if self.strategy.is_rank_0():
            return eval_loss_minimum

    def clean_old_checkpoints(self, output_dir, max_checkpoints=3):
        """Clean old checkpoints keeping only the most recent ones."""
        subdirs = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('step_')
        ]
        if len(subdirs) > max_checkpoints:
            subdirs.sort(key=lambda x: int(x.split('_')[1]))
            dir_to_delete = os.path.join(output_dir, subdirs[0])
            try:
                shutil.rmtree(dir_to_delete)
            except Exception as e:
                print(f"Error deleting old checkpoint {dir_to_delete}: {e}")
