# Borda Inflation in Parametric Reward Models on UltraFeedback

## Motivation

Siththaranjan et al. (2024) showed that BT reward is equivalent to Borda count. Parametric reward models share parameters across prompts, meaning the reward assigned to a response under prompt A is influenced by training data from prompt B. Per-prompt tabular BT-MLE is immune to this by construction (no parameter sharing).

A toy case on synthetic data (`condorcet_eval_aggregated.py`) confirmed that Borda inflation persists in neural embedding space: an aggregated RM trained on shared embeddings diverges from prompt-local BT-MLE. This experiment tests whether the same phenomenon holds on real LLM preference data.

## Setup

**Data**: openbmb/UltraFeedback, using all 4 evaluation dimensions (instruction_following, honesty, truthfulness, helpfulness) as proxy for diverse annotators. Each prompt has 4 responses scored on each dimension.

**Models** (trained on UltraFeedback multidim, 1 epoch, LLaMA-3-8B base):
- **RM** (dim=1): Standard scalar reward model. P_RM[i,j] = sigma((r_i - r_j) / tau).
- **PM** (dim=8): General Preference Model with skew-symmetric R matrix. P_PM[i,j] = sigma(v_i^T R^T v_j / tau). Capable of representing intransitive preferences.

**Eval splits** (fixed, reusable):
- **Seen**: 1024 prompts sampled from the training split
- **Unseen**: 1024 prompts sampled from the validation split

**Ground truth**: Majority-rule preference matrix P_emp[i,j] = fraction of 4 dimensions where score_i > score_j (ties count as 0.5).

**Methods compared** (5 ranking methods):
1. Emp.Borda: Borda scores from P_emp
2. Emp.Copeland: Copeland scores (pairwise victory count) from P_emp
3. Nash: Nash equilibrium (maximal lottery) from P_emp
4. RM.Borda: Borda scores from P_RM
5. PM.Borda: Borda scores from P_PM

## Results

### Calibration (Brier scores)

| Metric | Seen | Unseen |
|--------|------|--------|
| Brier RM (mean +/- std) | 0.0799 +/- 0.0599 | 0.0810 +/- 0.0630 |
| Brier PM (mean +/- std) | 0.0280 +/- 0.0247 | 0.0527 +/- 0.0493 |

PM pairwise probabilities are substantially better calibrated against empirical preferences on both splits. RM shows almost no seen/unseen gap (0.0799 vs 0.0810), while PM degrades more (0.0280 vs 0.0527), indicating partial memorization of training prompts.

### Winner Agreement

| Method pair | Seen | Unseen |
|-------------|------|--------|
| Emp.Borda - Emp.Copeland | 97.0% | 95.9% |
| Emp.Borda - Nash | 85.4% | 85.2% |
| Emp.Borda - RM.Borda | 54.2% | 49.3% |
| Emp.Borda - PM.Borda | 72.3% | 60.9% |
| RM.Borda - PM.Borda | 58.3% | 61.0% |

RM winner agreement with empirical ground truth is near chance (25% for K=4 uniform random, 50% suggests weak signal). PM is significantly better but drops from 72.3% to 60.9% on unseen data.

### Kendall Tau (full ranking correlation, per-prompt average)

| Method pair | Seen | Unseen |
|-------------|------|--------|
| Emp.Borda - Emp.Copeland | 0.963 | 0.958 |
| Emp.Borda - Nash | 0.528 | 0.536 |
| Emp.Borda - RM.Borda | 0.427 | 0.406 |
| Emp.Borda - PM.Borda | 0.746 | 0.586 |
| RM.Borda - Nash | 0.231 | 0.211 |
| PM.Borda - Nash | 0.405 | 0.306 |
| RM.Borda - PM.Borda | 0.483 | 0.529 |

PM tracks full rankings much better than RM (tau 0.746 vs 0.427 on seen). RM rankings are essentially uncorrelated with Nash equilibria (tau ~0.23).

### Cross-Prompt Inflation

| Metric | RM (seen) | PM (seen) | RM (unseen) | PM (unseen) |
|--------|-----------|-----------|-------------|-------------|
| inflation mean | 0.000 | 0.000 | 0.000 | 0.000 |
| inflation std | 1.122 | 0.679 | 1.163 | 0.923 |
| |inflation| mean | 0.764 | 0.361 | 0.794 | 0.574 |
| at rank 0 (no shift) | 44.9% | 68.7% | 44.0% | 54.9% |
| |shift| >= 2 | 17.8% | 4.6% | 18.8% | 10.6% |

Both RM and PM inflation distributions are perfectly symmetric around zero. There is no systematic directional bias — no evidence that either model consistently over- or under-ranks specific rank positions. This is expected: UltraFeedback responses come from different LLMs with no fixed ordering, so no single response position is consistently longer, more sycophantic, etc.

PM produces much less inflation than RM: on seen data, 68.7% of PM ranks exactly match empirical Borda (vs 44.9% for RM), and large shifts (|shift| >= 2) occur in only 4.6% of cases (vs 17.8% for RM).

### Winner Disagreement Breakdown

| Metric | Seen | Unseen |
|--------|------|--------|
| RM wrong | 45.8% (469) | 50.7% (519) |
| PM wrong | 27.7% (284) | 39.1% (400) |
| Both wrong | 19.2% (197) | 29.9% (306) |
| RM only wrong | 26.6% (272) | 20.8% (213) |
| PM only wrong | 8.5% (87) | 9.2% (94) |

When RM gets the winner wrong, PM rescues it 26.6% of the time (seen). PM-only errors are much rarer (8.5%). PM is strictly more informative than RM for winner selection on both splits.

### Logit-Transitivity and BT Distortion

| Metric | Seen | Unseen |
|--------|------|--------|
| LT violations PM (mean) | 0.196 | 0.164 |
| LT violations empirical (mean) | 0.996 | 0.984 |
| BT distortion of PM | 0.0029 | 0.0029 |
| Cycle rate (empirical) | 29.2% | 31.1% |

PM is nearly logit-transitive (mean violation ~0.2 vs ~1.0 for empirical), despite having the capacity for intransitive preferences (dim=8). The BT distortion of PM is negligible (0.003), meaning a BT model fits PM's preference matrix almost perfectly. This suggests that even though PM has dim=8, it effectively learns a near-transitive preference structure.

### RM vs Local BT-MLE

| Metric | Seen | Unseen |
|--------|------|--------|
| |P_RM - P_bt_local| (P-space) | 0.171 | 0.171 |
| ||r_RM - r_BT_MLE||_2 (reward-space) | 6.76 | 6.32 |

RM pairwise probabilities deviate from prompt-local BT-MLE by 0.17 on average, confirming cross-prompt parameter sharing distorts per-prompt reward structure. The reward-space L2 distance is large (~6.5), indicating RM's scalar rewards are on a very different scale from prompt-local BT log-strengths.

## Key Findings

1. **PM fits per-prompt preferences substantially better than RM**, with lower Brier scores, higher Kendall tau, higher winner agreement, and less inflation on both seen and unseen data.

2. **RM shows smaller seen/unseen gap than PM**, suggesting RM's poor performance is inherent to the scalar parameterization rather than overfitting. PM's advantage partially relies on memorizing training prompts.

3. **Both models show perfectly symmetric inflation** — no directional bias in rank shifts. Cross-prompt parameter sharing introduces noise-like disagreement with prompt-local rankings, not systematic over/under-valuation of specific response positions.

4. **PM is nearly logit-transitive** despite having capacity for intransitivity, and BT fits PM almost perfectly. The dim=8 representation doesn't meaningfully exploit intransitive preferences on this data.

5. **RM rankings are near-random** for winner selection (~50% agreement with Emp.Borda vs 25% chance) and essentially uncorrelated with Nash equilibria (tau ~0.23).

## Limitations

- Models trained for only 1 epoch; longer training may change the picture.
- F.normalize is commented out in the PM architecture (rw_model_general_preference.py), meaning PM vectors are not L2-normalized. This may affect PM's preference structure.
- Inflation analysis is at the rank/position level. Response-level behavioral features (length, sycophancy, hedging, etc.) are not yet analyzed — this is the subject of the next stage.

## Next Stage: Behavioral Analysis

The symmetric inflation distribution tells us *how much* parametric models disagree with prompt-local rankings, but not *what kind* of responses benefit or suffer. The next step uses OpenAI Batch API to annotate response features (length, sycophancy, hedging, specificity, etc.) and correlate them with inflation scores. This will reveal whether cross-prompt parameter sharing systematically amplifies specific behavioral patterns.

## Artifacts

- Eval data: `eval_1551803/eval_data/eval_{seen,unseen}.json`
- Inference: `eval_1551803/inference_{seen,unseen}.pkl`
- Analysis: `eval_1551803/analysis_{seen,unseen}/`
  - `summary.json`: Aggregate statistics
  - `results.pkl`: Full per-prompt data (P matrices, scores, rankings, inflation)
  - `flat.jsonl`: Per-response flat data for downstream analysis
  - `figure_*.png`: Winner agreement, calibration, inflation, logit-transitivity, RM vs BT
- Scripts: `experiments/borda_inflation/{prepare_eval_data,run_inference,analyze_inflation}.py`
- Cluster: `experiments/borda_inflation/{run_eval,submit_eval}.sh`

## Checkpoints

- RM: `llama3-8b-rm-20260217_112115_1534864/model_exports`
- PM: `llama3-8b-pm-20260217_112114_1534865/model_exports`
