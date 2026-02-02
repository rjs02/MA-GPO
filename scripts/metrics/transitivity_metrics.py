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
        all_loop_ratios = []
        all_spectral_gaps = []
        all_mfas_scores = []
        
        total_pairs = 0
        total_conflicts = 0
        total_cycles = 0
        
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
            
            # Count cycles
            if n_resp >= 3:
                n_cycles = self._count_triangles(pref_matrix)
                all_cycle_counts.append(n_cycles)
                total_cycles += n_cycles
            else:
                all_cycle_counts.append(0)
                
            # Hodge decomposition (if requested and feasible)
            loop_ratio = 0.0
            if compute_hodge and n_resp >= 3 and n_resp <= 100:
                try:
                    loop_ratio = self._compute_hodge_loop_ratio(pref_matrix)
                    all_loop_ratios.append(loop_ratio)
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Hodge computation failed for prompt: {e}")
                    all_loop_ratios.append(0.0)
            else:
                all_loop_ratios.append(0.0)
                
            # Spectral analysis
            spectral_gap = 0.0
            if compute_spectral and n_resp >= 3 and n_resp <= 100:
                try:
                    spectral_gap = self._compute_spectral_gap(pref_matrix)
                    all_spectral_gaps.append(spectral_gap)
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Spectral computation failed: {e}")
                    all_spectral_gaps.append(0.0)
            else:
                all_spectral_gaps.append(0.0)
                
            # MFAS score
            mfas_score = 0.0
            if compute_mfas and n_resp >= 3:
                try:
                    mfas_score = self._compute_mfas_score(pref_matrix)
                    all_mfas_scores.append(mfas_score)
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"MFAS computation failed: {e}")
                    all_mfas_scores.append(0.0)
            else:
                all_mfas_scores.append(0.0)
                
            # Store per-prompt stats
            self.per_prompt_stats[prompt[:100]] = {
                "n_responses": n_resp,
                "n_pairs": n_pairs,
                "conflict_rate": conflict_rate,
                "n_cycles": n_cycles if n_resp >= 3 else 0,
                "loop_ratio": loop_ratio,
                "spectral_gap": spectral_gap,
                "mfas_score": mfas_score,
            }
            
        # Aggregate statistics (convert numpy types to Python types for JSON serialization)
        results = {
            "total_pairs": int(total_pairs),
            "unique_prompts": int(len(prompt_groups)),
            "conflict_rate": float(total_conflicts / total_pairs if total_pairs > 0 else 0.0),
            "triangle_cycles": int(total_cycles),
            "mean_conflict_rate": float(np.mean(all_conflict_rates) if all_conflict_rates else 0.0),
            "mean_cycles_per_prompt": float(np.mean(all_cycle_counts) if all_cycle_counts else 0.0),
            "mean_loop_ratio_hodge": float(np.mean([x for x in all_loop_ratios if x > 0 and not np.isnan(x)]) if all_loop_ratios else 0.0),
            "mean_spectral_gap": float(np.mean([x for x in all_spectral_gaps if x > 0 and not np.isnan(x)]) if all_spectral_gaps else 0.0),
            "mean_mfas_score": float(np.mean([x for x in all_mfas_scores if x > 0 and not np.isnan(x)]) if all_mfas_scores else 0.0),
            "per_prompt_stats_sample": dict(list(self.per_prompt_stats.items())[:10]),
            "prompts_with_multiple_responses": int(sum(1 for stats in self.per_prompt_stats.values() if stats["n_responses"] >= 3)),
            "prompts_with_conflicts": int(sum(1 for stats in self.per_prompt_stats.values() if stats["conflict_rate"] > 0)),
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
        Count 3-cycles (A>B, B>C, C>A).
        
        Uses directed triangle counting algorithm.
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
    print(f"Total triangle cycles: {results['triangle_cycles']:,}")
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
