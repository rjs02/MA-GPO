# Borda Inflation Experiment

Measures whether parametric reward models (BT RM dim=1, PM dim=8) trained on UltraFeedback exhibit cross-prompt Borda inflation compared to per-prompt tabular BT-MLE ground truth.

## Pipeline Overview

```
[UltraFeedback data]
        |
        v
  prepare_eval_data.py        # Fixed eval splits (1024 prompts × 2 splits)
        |
        v
  run_inference.py             # GPU: RM/PM forward passes → inference pickle
        |                       (submit_eval.sh / run_eval.sh on SLURM)
        v
  analyze_inflation.py         # CPU: social choice metrics, figures, flat.jsonl
        |
        v
  extract_local_features.py    # Merge eval data + flat.jsonl + 12 programmatic features
        |                       → features.jsonl (8192 entries)
        v
  analyze_behavioral.py        # Correlations: local features × inflation
        |
        v
  epistemic_labeler.py         # GPT-5.2 Batch API: 7-category epistemic strategy labels
        |                       Uses openai_batch_api.py (supervisor's helper class)
        v
  parse_gpt_results.py         # Parse batch output, merge → features_labeled.jsonl
        |
        v
  analyze_strategy_game.py     # Aggregate to 7×7/3×3 strategy-level preference matrices
        |                       Borda scores, logit-transitivity, prompt-conditioned analysis
        v
  analyze_epistemic.py         # Per-response: inflation by strategy, substance vs surface
```

## Key Files

### Stage 1: Reward Model Evaluation (cluster GPU)

| File | Purpose |
|------|---------|
| `prepare_eval_data.py` | Create fixed eval splits from UltraFeedback (CPU, one-time) |
| `run_inference.py` | RM/PM forward passes on eval splits (2× A100 GPUs) |
| `run_eval.sh` | Orchestrator: data prep → inference → analysis |
| `submit_eval.sh` | SLURM wrapper (4h, 1 node, 2 GPUs, account a166) |
| `tabular_inflation.py` | Reusable social choice functions (Borda, Copeland, Nash, BT-MLE) |
| `analyze_inflation.py` | Compute all metrics + figures from inference results (CPU) |

### Stage 2a: Local Behavioral Features (CPU)

| File | Purpose |
|------|---------|
| `behavioral_features/extract_local_features.py` | Compute 12 features (word count, formatting, etc.) |
| `behavioral_features/analyze_behavioral.py` | Correlations: features × inflation |

### Stage 2b: GPT Epistemic Strategy Classification

| File | Purpose |
|------|---------|
| `behavioral_features/openai_batch_api.py` | Supervisor's `OpenAIBatchAPIHelper` base class |
| `behavioral_features/epistemic_labeler.py` | Subclass: prepare/launch/retrieve batch jobs (gpt-5.2) |
| `behavioral_features/parse_gpt_results.py` | Parse batch output, merge labels, class distribution |
| `behavioral_features/label_viewer.html` | Browser-based label inspector (drag-drop JSONL) |

### Stage 2c: Strategy-Level Game Analysis

| File | Purpose |
|------|---------|
| `behavioral_features/analyze_strategy_game.py` | Aggregate 4×4 → 7×7/3×3 preference matrices |
| `behavioral_features/analyze_epistemic.py` | Per-response inflation × strategy correlations |

### Already-Tracked Dependencies (not modified)

| File | Purpose |
|------|---------|
| `eval/evaluate_trained_models_ultrafeedback.py` | `load_reward_model()`, `get_response_reward()`, etc. |
| `general_preference/models/rw_model_general_preference.py` | PM architecture (R matrix, skew-symmetric) |

### Report

| File | Purpose |
|------|---------|
| `REPORT_parametric_inflation.md` | Stage 1 results write-up |

## Data Flow

**Eval data** (on cluster): `/home/robin/eth/CLARIDEN/SAE_Borda_inflation/eval_1551803/`
```
eval_data/eval_seen.json       # 1024 prompts × 4 responses (training split)
eval_data/eval_unseen.json     # 1024 prompts × 4 responses (val split)
inference_seen.pkl             # RM rewards + PM preference matrices
inference_unseen.pkl
analysis_seen/results.pkl      # Full per-prompt results (P_emp, P_RM, P_PM, etc.)
analysis_seen/flat.jsonl       # Per-response flat data
analysis_seen/summary.json
analysis_unseen/...
```

**Behavioral features** (local): `behavioral_features/`
```
features.jsonl                 # 8192 entries: response text + Stage 1 metrics + 12 local features
features_pilot_labeled.jsonl   # Pilot (300 entries labeled)
results_eval_8191_v1/          # Full labeling run (8191 entries)
  batch_output_0.jsonl         # Raw GPT-5.2 output
  features_eval_v1.jsonl       # Fully labeled dataset
  strategy_game_summary.json   # 7×7 and 3×3 aggregated analysis
  figure_*.png                 # All figures
```

## Key Configs

- **RM**: dim=1, tau=1.0 (Bradley-Terry scalar reward)
- **PM**: dim=8, tau=0.1 (General Preference Model, skew-symmetric R matrix)
- **Eval**: 1024 prompts × 4 responses × 2 splits = 8192 total responses
- **GPT labeling**: gpt-5.2, temperature=0.0, JSON mode, 7-category epistemic strategy scheme
- **Models trained on**: Full UltraFeedback training set (multidimensional, no conflict removal)

## 7 Epistemic Strategy Categories

1. **Genuinely helpful** — substantive, accurate, directly engages
2. **Sycophantic** — flattery, agreement over accuracy
3. **Superficially polished** — elaborate formatting, low substance-to-length ratio
4. **Assertive confabulation** — confident but potentially inaccurate claims
5. **Hedging/overcautious** — excessive caveats, refusal on benign queries
6. **Evasive/deflective** — redirects, generic advice, topic-shifting
7. **Concise and substantive** — short, high information density

Binary grouping: Substance (1,7) / Surface (2,3,4) / Neither (5,6)
