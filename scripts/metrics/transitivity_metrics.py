#!/usr/bin/env python3
"""
Core module for quantifying intransitivity in preference datasets.

Implements:
- Cycle counting (triangles/3-cycles)
- Hodge decomposition (gradient, curl, harmonic components)
- Spectral analysis (eigenvalue-based cycle detection)
- MFAS approximation (minimum feedback arc set)

All algorithms are optimized for large-scale datasets (100k+ preference pairs).
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
from scipy.linalg import eigh
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings


class TransitivityAnalyzer:
    """
    Analyzer for quantifying intransitivity in preference datasets.
    
    Operates on per-prompt basis to avoid O(n²) explosion when n is large.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.per_prompt_stats = {}
        
    def analyze_dataset(
        self,
        preferences: List[Tuple[str, str, str]],
        max_samples: Optional[int] = None,
        compute_hodge: bool = True,
        compute_spectral: bool = True,
        compute_mfas: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze intransitivity in a preference dataset.
        
        Args:
            preferences: List of (prompt, chosen, rejected) tuples
            max_samples: Limit analysis to first N samples (for speed)
            compute_hodge: Whether to compute Hodge decomposition
            compute_spectral: Whether to compute spectral metrics
            compute_mfas: Whether to compute MFAS
            
        Returns:
            Dictionary with transitivity metrics
        """
        if max_samples:
            preferences = preferences[:max_samples]
            
        if self.verbose:
            print(f"Analyzing {len(preferences)} preference pairs...")
            
        # Group by prompt
        prompt_groups = self._group_by_prompt(preferences)
        
        if self.verbose:
            print(f"Found {len(prompt_groups)} unique prompts")
            
        # Compute metrics per prompt
        all_conflict_rates = []
        all_cycle_counts = []
        all_instance_cycle_counts = []
        all_loop_ratios = []
        all_adjusted_loop_ratios = []
        all_baseline_loop_ratios = []
        all_spectral_gaps = []
        all_mfas_scores = []
        
        total_pairs = 0
        total_conflicts = 0
        total_cycles = 0
        total_instance_cycles = 0
        
        iterator = tqdm(prompt_groups.items(), desc="Analyzing prompts") if self.verbose else prompt_groups.items()
        
        for prompt, pairs in iterator:
            # Build response set and adjacency
            responses = self._extract_responses(pairs)
            n_resp = len(responses)
            
            if n_resp < 2:
                continue
                
            # Build preference matrix (win counts)
            pref_matrix = self._build_preference_matrix(pairs, responses)
            n_pairs = np.sum(pref_matrix > 0)
            total_pairs += n_pairs
            
            # Basic conflict rate
            conflict_rate = self._compute_conflict_rate(pref_matrix)
            all_conflict_rates.append(conflict_rate)
            n_conflicts = int(conflict_rate * n_pairs)
            total_conflicts += n_conflicts
            
            # Count cycles (majority-vote aggregated)
            n_cycles = 0
            n_instance_cycles = 0
            if n_resp >= 3:
                n_cycles = self._count_triangles(pref_matrix)
                all_cycle_counts.append(n_cycles)
                total_cycles += n_cycles
                
                # Count cycles (instance-level)
                n_instance_cycles = self._count_instance_triangles(pairs, responses)
                all_instance_cycle_counts.append(n_instance_cycles)
                total_instance_cycles += n_instance_cycles
            else:
                all_cycle_counts.append(0)
                all_instance_cycle_counts.append(0)
                
            # Hodge decomposition (limit to <=200 for computational efficiency)
            loop_ratio = np.nan
            baseline_loop_ratio = np.nan
            adjusted_loop_ratio = np.nan
            if compute_hodge and n_resp >= 3 and n_resp <= 200:
                try:
                    loop_ratio = self._compute_hodge_loop_ratio(pref_matrix)
                    all_loop_ratios.append(loop_ratio)
                    
                    # Compute baseline (structural floor due to observation topology)
                    baseline_loop_ratio = self._compute_transitive_baseline(pref_matrix)
                    all_baseline_loop_ratios.append(baseline_loop_ratio)
                    
                    # Adjusted loop ratio = (observed - baseline) / (1 - baseline)
                    # This removes the structural baseline and normalizes to [0, 1]
                    if baseline_loop_ratio < 1.0 - 1e-10:
                        adjusted_loop_ratio = (loop_ratio - baseline_loop_ratio) / (1.0 - baseline_loop_ratio)
                        adjusted_loop_ratio = float(np.clip(adjusted_loop_ratio, 0, 1))
                    else:
                        adjusted_loop_ratio = 0.0
                    all_adjusted_loop_ratios.append(adjusted_loop_ratio)
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Hodge computation failed for prompt: {e}")
                    all_loop_ratios.append(np.nan)
                    all_baseline_loop_ratios.append(np.nan)
                    all_adjusted_loop_ratios.append(np.nan)
            else:
                all_loop_ratios.append(np.nan)
                all_baseline_loop_ratios.append(np.nan)
                all_adjusted_loop_ratios.append(np.nan)
                
            # Spectral analysis (limit to <=200 for computational efficiency)
            spectral_gap = np.nan
            if compute_spectral and n_resp >= 3 and n_resp <= 200:
                try:
                    spectral_gap = self._compute_spectral_gap(pref_matrix)
                    all_spectral_gaps.append(spectral_gap)
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Spectral computation failed: {e}")
                    all_spectral_gaps.append(np.nan)
            else:
                all_spectral_gaps.append(np.nan)
                
            # MFAS score
            mfas_score = np.nan
            if compute_mfas and n_resp >= 3:
                try:
                    mfas_score = self._compute_mfas_score(pref_matrix)
                    all_mfas_scores.append(mfas_score)
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"MFAS computation failed: {e}")
                    all_mfas_scores.append(np.nan)
            else:
                all_mfas_scores.append(np.nan)
                
            # Store per-prompt stats
            self.per_prompt_stats[prompt[:100]] = {
                "n_responses": n_resp,
                "n_pairs": n_pairs,
                "conflict_rate": conflict_rate,
                "n_cycles_majority": n_cycles if n_resp >= 3 else 0,
                "n_cycles_instance": n_instance_cycles if n_resp >= 3 else 0,
                "loop_ratio": float(loop_ratio) if not np.isnan(loop_ratio) else None,
                "baseline_loop_ratio": float(baseline_loop_ratio) if not np.isnan(baseline_loop_ratio) else None,
                "adjusted_loop_ratio": float(adjusted_loop_ratio) if not np.isnan(adjusted_loop_ratio) else None,
                "spectral_gap": float(spectral_gap) if not np.isnan(spectral_gap) else None,
                "mfas_score": float(mfas_score) if not np.isnan(mfas_score) else None,
            }
            
        # Count how many prompts each metric was computed for
        n_hodge_computed = int(np.sum(~np.isnan(all_loop_ratios)))
        n_spectral_computed = int(np.sum(~np.isnan(all_spectral_gaps)))
        n_mfas_computed = int(np.sum(~np.isnan(all_mfas_scores)))
        
        # Compute response count distribution
        response_counts = [stats["n_responses"] for stats in self.per_prompt_stats.values()]
        response_count_dist = {
            "min": int(np.min(response_counts)) if response_counts else 0,
            "25th": int(np.percentile(response_counts, 25)) if response_counts else 0,
            "median": int(np.median(response_counts)) if response_counts else 0,
            "75th": int(np.percentile(response_counts, 75)) if response_counts else 0,
            "90th": int(np.percentile(response_counts, 90)) if response_counts else 0,
            "95th": int(np.percentile(response_counts, 95)) if response_counts else 0,
            "99th": int(np.percentile(response_counts, 99)) if response_counts else 0,
            "max": int(np.max(response_counts)) if response_counts else 0,
            "mean": float(np.mean(response_counts)) if response_counts else 0.0,
            "n_over_200": int(np.sum(np.array(response_counts) > 200)),
        }
        
        # Aggregate statistics (convert numpy types to Python types for JSON serialization)
        results = {
            "total_pairs": int(total_pairs),
            "unique_prompts": int(len(prompt_groups)),
            "conflict_rate": float(total_conflicts / total_pairs if total_pairs > 0 else 0.0),
            "triangle_cycles_majority": int(total_cycles),
            "triangle_cycles_instance": int(total_instance_cycles),
            "mean_conflict_rate": float(np.mean(all_conflict_rates) if all_conflict_rates else 0.0),
            "mean_cycles_per_prompt_majority": float(np.mean(all_cycle_counts) if all_cycle_counts else 0.0),
            "mean_cycles_per_prompt_instance": float(np.mean(all_instance_cycle_counts) if all_instance_cycle_counts else 0.0),
            "mean_loop_ratio_hodge": float(np.nanmean(all_loop_ratios) if all_loop_ratios else 0.0),
            "mean_baseline_loop_ratio": float(np.nanmean(all_baseline_loop_ratios) if all_baseline_loop_ratios else 0.0),
            "mean_adjusted_loop_ratio": float(np.nanmean(all_adjusted_loop_ratios) if all_adjusted_loop_ratios else 0.0),
            "mean_spectral_gap": float(np.nanmean(all_spectral_gaps) if all_spectral_gaps else 0.0),
            "mean_mfas_score": float(np.nanmean(all_mfas_scores) if all_mfas_scores else 0.0),
            "n_hodge_computed": n_hodge_computed,
            "n_spectral_computed": n_spectral_computed,
            "n_mfas_computed": n_mfas_computed,
            "per_prompt_stats_sample": dict(list(self.per_prompt_stats.items())[:10]),
            "prompts_with_multiple_responses": int(sum(1 for stats in self.per_prompt_stats.values() if stats["n_responses"] >= 3)),
            "prompts_with_conflicts": int(sum(1 for stats in self.per_prompt_stats.values() if stats["conflict_rate"] > 0)),
            "response_count_distribution": response_count_dist,
        }
        
        # Add distributions
        results["conflict_rate_distribution"] = {
            "min": float(np.min(all_conflict_rates)) if all_conflict_rates else 0.0,
            "25th": float(np.percentile(all_conflict_rates, 25)) if all_conflict_rates else 0.0,
            "median": float(np.median(all_conflict_rates)) if all_conflict_rates else 0.0,
            "75th": float(np.percentile(all_conflict_rates, 75)) if all_conflict_rates else 0.0,
            "max": float(np.max(all_conflict_rates)) if all_conflict_rates else 0.0,
        }
        
        return results
    
    def _group_by_prompt(self, preferences: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        """Group preferences by prompt."""
        groups = defaultdict(list)
        for prompt, chosen, rejected in preferences:
            groups[prompt].append((chosen, rejected))
        return dict(groups)
    
    def _extract_responses(self, pairs: List[Tuple[str, str]]) -> List[str]:
        """Extract unique responses from preference pairs."""
        responses = set()
        for chosen, rejected in pairs:
            responses.add(chosen)
            responses.add(rejected)
        return sorted(list(responses))
    
    def _build_preference_matrix(
        self, 
        pairs: List[Tuple[str, str]], 
        responses: List[str]
    ) -> np.ndarray:
        """
        Build preference matrix where M[i,j] = number of times i was preferred over j.
        """
        n = len(responses)
        resp_to_idx = {r: i for i, r in enumerate(responses)}
        matrix = np.zeros((n, n), dtype=float)
        
        for chosen, rejected in pairs:
            i = resp_to_idx[chosen]
            j = resp_to_idx[rejected]
            matrix[i, j] += 1
            
        return matrix
    
    def _compute_conflict_rate(self, pref_matrix: np.ndarray) -> float:
        """
        Compute fraction of pairs with conflicts (A>B and B>A both exist).
        """
        n = pref_matrix.shape[0]
        total_pairs = 0
        conflicting_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if pref_matrix[i, j] > 0 or pref_matrix[j, i] > 0:
                    total_pairs += 1
                    if pref_matrix[i, j] > 0 and pref_matrix[j, i] > 0:
                        conflicting_pairs += 1
                        
        return conflicting_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _count_triangles(self, pref_matrix: np.ndarray) -> int:
        """
        Count 3-cycles (A>B, B>C, C>A) in the MAJORITY-VOTE aggregated graph.
        
        Uses directed triangle counting algorithm on the aggregated preference matrix.
        """
        n = pref_matrix.shape[0]
        cycles = 0
        
        # Build adjacency (A[i,j] = 1 if i preferred over j more often)
        adj = (pref_matrix > pref_matrix.T).astype(int)
        
        # Count triangles: for each triplet, check if cycle exists
        for i in range(n):
            for j in range(n):
                if i != j and adj[i, j]:
                    for k in range(n):
                        if k != i and k != j and adj[j, k] and adj[k, i]:
                            cycles += 1
                            
        # Each cycle counted once (i->j->k->i)
        return cycles
    
    def _count_instance_triangles(self, pairs: List[Tuple[str, str]], responses: List[str]) -> int:
        """
        Count 3-cycles at the INSTANCE level (raw preference pairs).
        
        For each triplet of responses (A, B, C), counts how many times we observe
        the cycle A>B, B>C, C>A in the raw preference data.
        
        This counts MANY more cycles than majority-vote method since it looks at
        individual preference instances rather than aggregated wins/losses.
        """
        n = len(responses)
        resp_to_idx = {r: i for i, r in enumerate(responses)}
        
        # Build set of directed edges from raw pairs
        edges = set()
        for chosen, rejected in pairs:
            i = resp_to_idx[chosen]
            j = resp_to_idx[rejected]
            edges.add((i, j))
        
        # Count cycles: for each ordered triplet, check if all three edges exist
        cycles = 0
        for i in range(n):
            for j in range(n):
                if i != j and (i, j) in edges:
                    for k in range(n):
                        if k != i and k != j and (j, k) in edges and (k, i) in edges:
                            cycles += 1
        
        return cycles
    
    def _compute_hodge_loop_ratio(self, pref_matrix: np.ndarray) -> float:
        """
        Compute loop ratio using Hodge decomposition.
        
        Loop ratio η_L = (||X_C||² + ||X_H||²) / ||X||²
        where X is the preference flow, X_G is gradient component,
        and X_C + X_H are curl + harmonic (cyclic) components.
        """
        n = pref_matrix.shape[0]
        
        # Build preference flow: X[i,j] = win_rate(i,j) - 0.5
        total_comparisons = pref_matrix + pref_matrix.T
        with np.errstate(divide='ignore', invalid='ignore'):
            win_rates = np.where(total_comparisons > 0, pref_matrix / total_comparisons, 0.5)
        flow = win_rates - 0.5
        
        # Compute total energy
        total_energy = np.sum(flow ** 2)
        
        if total_energy < 1e-10:
            return 0.0
        
        # Compute gradient component via least squares
        # Find potential r that minimizes ||flow - gradient(r)||²
        # gradient(r)[i,j] = r[j] - r[i]
        
        # Build system: A @ r = b where we minimize ||flow - (r_j - r_i)||²
        # This is equivalent to graph Laplacian system
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i != j and total_comparisons[i, j] > 0:
                    A[i, i] += 1
                    A[j, j] += 1
                    A[i, j] -= 1
                    A[j, i] -= 1
                    b[i] += flow[i, j]
                    b[j] -= flow[i, j]
        
        # Fix one node to remove translation invariance
        A[0, :] = 0
        A[0, 0] = 1
        b[0] = 0
        
        try:
            r = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Singular matrix, use pseudoinverse
            r = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Compute gradient flow
        gradient_flow = np.zeros_like(flow)
        for i in range(n):
            for j in range(n):
                if i != j:
                    gradient_flow[i, j] = r[j] - r[i]
        
        # Gradient energy
        gradient_energy = np.sum(gradient_flow ** 2)
        
        # Loop energy = total - gradient
        loop_energy = max(0, total_energy - gradient_energy)
        
        loop_ratio = loop_energy / total_energy if total_energy > 0 else 0.0
        
        return float(np.clip(loop_ratio, 0, 1))
    
    def _compute_transitive_baseline(self, pref_matrix: np.ndarray) -> float:
        """
        Compute structural baseline loop ratio for a perfectly transitive graph.
        
        Takes the observed edge structure and enforces perfect transitivity based
        on Bradley-Terry scores, then computes Hodge decomposition. This gives
        the structural baseline loop ratio (floor due to observation topology).
        
        Returns:
            Baseline loop ratio (0 = graph topology is perfectly transitive)
        """
        n = pref_matrix.shape[0]
        
        # Compute Bradley-Terry scores via Laplacian solver
        total_comparisons = pref_matrix + pref_matrix.T
        with np.errstate(divide='ignore', invalid='ignore'):
            win_rates = np.where(total_comparisons > 0, pref_matrix / total_comparisons, 0.5)
        
        # Build Laplacian system to get BT scores
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i != j and total_comparisons[i, j] > 0:
                    A[i, i] += 1
                    A[j, j] += 1
                    A[i, j] -= 1
                    A[j, i] -= 1
                    b[i] += win_rates[i, j] - 0.5
                    b[j] -= win_rates[i, j] - 0.5
        
        # Fix one node
        A[0, :] = 0
        A[0, 0] = 1
        b[0] = 0
        
        try:
            bt_scores = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            bt_scores = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Create synthetic transitive preference matrix using BT scores
        # but with the same edge topology as the observed data
        synthetic = np.zeros_like(pref_matrix)
        for i in range(n):
            for j in range(n):
                if total_comparisons[i, j] > 0:
                    # Enforce transitive ordering: higher BT score always wins
                    if bt_scores[i] > bt_scores[j]:
                        synthetic[i, j] = 1
                    elif bt_scores[j] > bt_scores[i]:
                        synthetic[j, i] = 1
                    else:
                        # Tie: assign randomly but consistently
                        synthetic[i, j] = 0.5
        
        # Compute loop ratio on synthetic transitive graph
        baseline_loop_ratio = self._compute_hodge_loop_ratio(synthetic)
        
        return baseline_loop_ratio
    
    def _compute_spectral_gap(self, pref_matrix: np.ndarray) -> float:
        """
        Compute spectral gap of the preference graph.
        
        Transitive graphs have large spectral gap.
        Cyclic graphs have small gap and complex eigenvalues.
        """
        n = pref_matrix.shape[0]
        
        # Build skew-symmetric part: S = (A - A^T) / 2
        adj = (pref_matrix > pref_matrix.T).astype(float)
        skew = (adj - adj.T) / 2.0
        
        try:
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(skew)
            
            # Sort by absolute value
            sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
            
            # Spectral gap is difference between largest and second largest
            if len(sorted_eigs) >= 2:
                gap = float(sorted_eigs[0] - sorted_eigs[1])
            else:
                gap = 0.0
                
            return gap
        except np.linalg.LinAlgError:
            return 0.0
    
    def _compute_mfas_score(self, pref_matrix: np.ndarray) -> float:
        """
        Compute approximate MFAS (Minimum Feedback Arc Set) score.
        
        Uses greedy algorithm: repeatedly remove node with smallest
        |in_degree - out_degree| until no cycles remain.
        
        Returns: fraction of edges to remove for acyclicity (0 = transitive, 1 = fully cyclic)
        """
        n = pref_matrix.shape[0]
        
        # Build directed adjacency (i->j if i preferred over j)
        adj = (pref_matrix > pref_matrix.T).astype(int)
        total_edges = int(np.sum(adj))
        
        if total_edges == 0:
            return 0.0
        
        # Greedy MFAS approximation
        adj_copy = adj.copy()
        ordering = []
        removed_edges = 0
        
        max_iterations = n * 2  # Prevent infinite loops
        iteration = 0
        
        while adj_copy.sum() > 0 and iteration < max_iterations:
            iteration += 1
            
            # Compute in/out degrees
            in_deg = np.sum(adj_copy, axis=0)
            out_deg = np.sum(adj_copy, axis=1)
            
            # Find node with minimum |in - out|
            diff = np.abs(in_deg - out_deg)
            # Only consider nodes with edges
            mask = (in_deg + out_deg) > 0
            if not np.any(mask):
                break
            
            # Set infinite for nodes already processed
            diff = diff.astype(float)
            diff[~mask] = np.inf
            
            if np.all(np.isinf(diff)):
                break
                
            node = int(np.argmin(diff))
            
            # Add to ordering
            ordering.append(node)
            
            # Count backward edges that need to be removed
            for prev_node in ordering[:-1]:
                if adj_copy[node, prev_node] > 0:
                    removed_edges += 1
            
            # Remove all edges from/to this node
            adj_copy[node, :] = 0
            adj_copy[:, node] = 0
        
        mfas_score = removed_edges / total_edges if total_edges > 0 else 0.0
        
        return float(np.clip(mfas_score, 0, 1))


def load_preferences_from_jsonl(
    filepath: str,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """
    Load preferences from JSONL file.
    
    Args:
        filepath: Path to JSONL file
        prompt_key: Key for prompt field
        chosen_key: Key for chosen response
        rejected_key: Key for rejected response
        max_samples: Limit to first N samples
        
    Returns:
        List of (prompt, chosen, rejected) tuples
    """
    import json
    
    preferences = []
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
                
            data = json.loads(line)
            
            # Extract text content from structured format
            prompt = data[prompt_key]
            if isinstance(prompt, list):
                prompt = prompt[0].get('content', str(prompt[0]))
            
            chosen = data[chosen_key]
            if isinstance(chosen, list):
                chosen = chosen[0].get('content', str(chosen[0]))
                
            rejected = data[rejected_key]
            if isinstance(rejected, list):
                rejected = rejected[0].get('content', str(rejected[0]))
            
            preferences.append((str(prompt), str(chosen), str(rejected)))
    
    return preferences


def load_preferences_from_dataset(
    dataset,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """
    Load preferences from HuggingFace Dataset object.
    
    Args:
        dataset: HuggingFace Dataset
        prompt_key: Key for prompt field
        chosen_key: Key for chosen response
        rejected_key: Key for rejected response
        max_samples: Limit to first N samples
        
    Returns:
        List of (prompt, chosen, rejected) tuples
    """
    preferences = []
    
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    for i in range(n):
        sample = dataset[i]
        
        # Extract text content
        prompt = sample[prompt_key]
        if isinstance(prompt, list):
            prompt = prompt[0].get('content', str(prompt[0]))
        
        chosen = sample[chosen_key]
        if isinstance(chosen, list):
            chosen = chosen[0].get('content', str(chosen[0]))
            
        rejected = sample[rejected_key]
        if isinstance(rejected, list):
            rejected = rejected[0].get('content', str(rejected[0]))
        
        preferences.append((str(prompt), str(chosen), str(rejected)))
    
    return preferences


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze transitivity of preference dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL file or HF dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit analysis to N samples")
    parser.add_argument("--output", type=str, default="transitivity_report.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load preferences
    print(f"Loading preferences from {args.dataset}...")
    if args.dataset.endswith('.jsonl'):
        preferences = load_preferences_from_jsonl(args.dataset, max_samples=args.max_samples)
    else:
        from datasets import load_from_disk
        dataset = load_from_disk(args.dataset)
        preferences = load_preferences_from_dataset(dataset, max_samples=args.max_samples)
    
    # Analyze
    analyzer = TransitivityAnalyzer(verbose=True)
    results = analyzer.analyze_dataset(preferences)
    
    # Print results
    print("\n" + "="*60)
    print("TRANSITIVITY ANALYSIS RESULTS")
    print("="*60)
    print(f"Total pairs: {results['total_pairs']:,}")
    print(f"Unique prompts: {results['unique_prompts']:,}")
    print(f"Overall conflict rate: {results['conflict_rate']:.4f}")
    print(f"Total triangle cycles (majority-vote): {results['triangle_cycles_majority']:,}")
    print(f"Total triangle cycles (instance-level): {results['triangle_cycles_instance']:,}")
    print(f"Mean loop ratio (Hodge): {results['mean_loop_ratio_hodge']:.4f}")
    print(f"Mean spectral gap: {results['mean_spectral_gap']:.4f}")
    print(f"Mean MFAS score: {results['mean_mfas_score']:.4f}")
    print("\nConflict rate distribution:")
    for k, v in results['conflict_rate_distribution'].items():
        print(f"  {k}: {v:.4f}")
    
    # Save to file
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
