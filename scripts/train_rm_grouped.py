# Training script for GPM with grouped data format
#
# This script uses the optimized grouped data format where all responses
# for a prompt are processed together, reducing forward passes from O(K²)
# to O(K) per prompt.
#
# Usage:
#   deepspeed --num_gpus=2 scripts/train_rm_grouped.py \
#       --pretrain Qwen/Qwen3-0.6B \
#       --dataset data/ultrafeedback_grouped/pref_grouped_train \
#       --eval_dataset data/ultrafeedback_grouped/pref_grouped_val \
#       --is_general_preference \
#       --value_head_dim 6 \
#       ...

import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from general_preference.datasets import GroupedRewardDatasetV2
from general_preference.models import get_reward_model
from general_preference.trainer import GroupedPreferenceRewardTrainer
from general_preference.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    # Configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # Configure model
    model = get_reward_model(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(),
        init_value_head=True,
        is_general_preference=args.is_general_preference,
        value_head_dim=args.value_head_dim,
        init_prompt_head=True,
        add_prompt_head=args.add_prompt_head,
    )

    # Configure tokenizer
    tokenizer = get_tokenizer(
        args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer
    )
    tokenizer.truncation_side = "right"

    strategy.print(model)
    strategy.print(f"\n=== Grouped Training Mode ===")
    strategy.print(f"Using optimized O(K) forward passes per prompt")

    # Configure optimizer
    optim = strategy.create_optimizer(
        model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
    )

    # Load grouped datasets
    strategy.print(f"\nLoading grouped training data from: {args.dataset}")
    train_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        stopping_strategy="all_exhausted",
    )

    # Create grouped dataset
    train_dataset = GroupedRewardDatasetV2(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        max_comparisons_per_entry=args.max_comparisons_per_entry,
    )

    # For grouped data, we typically use batch_size=1 at the entry level
    # since each entry already contains multiple responses
    # The effective batch size comes from gradient accumulation
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,  # Often 1 for grouped format
        pin_memory=True,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    # Evaluation dataset (optional)
    eval_dataloader = None
    if args.eval_dataset:
        strategy.print(f"Loading grouped eval data from: {args.eval_dataset}")
        max_eval = args.max_eval_samples if args.max_eval_samples else args.max_samples
        eval_data = blending_datasets(
            args.eval_dataset,
            "1.0",
            strategy,
            args.seed,
            max_count=max_eval,
            stopping_strategy="all_exhausted",
        )
        eval_dataset = GroupedRewardDatasetV2(
            eval_data,
            tokenizer,
            args.max_len,
            strategy,
            max_comparisons_per_entry=args.max_comparisons_per_entry,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_train_batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn,
        )
    elif args.train_split_ratio < 1:
        # Split training data for validation
        strategy.print(f"Using {1 - args.train_split_ratio:.0%} of training data for validation")
        total_len = len(train_data)
        train_len = int(total_len * args.train_split_ratio)

        train_data_split = train_data.select(range(train_len))
        eval_data_split = train_data.select(range(train_len, total_len))

        train_dataset = GroupedRewardDatasetV2(
            train_data_split,
            tokenizer,
            args.max_len,
            strategy,
            max_comparisons_per_entry=args.max_comparisons_per_entry,
        )
        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            args.micro_train_batch_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )

        if len(eval_data_split) > 0:
            eval_dataset = GroupedRewardDatasetV2(
                eval_data_split,
                tokenizer,
                args.max_len,
                strategy,
            )
            eval_dataloader = strategy.setup_dataloader(
                eval_dataset,
                args.micro_train_batch_size,
                pin_memory=True,
                shuffle=False,
                collate_fn=eval_dataset.collate_fn,
            )

    # Scheduler
    num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.warmup_ratio),
        num_training_steps=max_steps,
    )

    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # Strategy prepare
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    os.makedirs(args.save_path, exist_ok=True)

    # Print training stats
    strategy.print(f"\n=== Training Configuration ===")
    strategy.print(f"Train entries: {len(train_dataset)}")
    strategy.print(f"Eval entries: {len(eval_dataloader.dataset) if eval_dataloader else 0}")
    strategy.print(f"Batch size (entries): {args.micro_train_batch_size}")
    strategy.print(f"Gradient accumulation: {args.accumulated_gradient}")
    strategy.print(f"Max epochs: {args.max_epochs}")
    strategy.print(f"Steps per epoch: {num_update_steps_per_epoch}")
    strategy.print(f"Total steps: {max_steps}")
    strategy.print(f"Model type: {'GPM' if args.is_general_preference else 'Bradley-Terry'}")
    strategy.print(f"Value head dim: {args.value_head_dim}")
    strategy.print(f"Margin loss: {args.margin_loss}")

    # Create grouped trainer
    trainer = GroupedPreferenceRewardTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_epochs=args.max_epochs,
        tau=args.general_preference_tau,
        value_head_dim=args.value_head_dim,
        is_general_preference=args.is_general_preference,
    )

    trainer.fit(args)

    # Save final model
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GPM reward model with grouped data format"
    )

    # Model arguments
    parser.add_argument("--pretrain", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--is_general_preference", action="store_true", default=False,
                       help="Use GPM (General Preference Model). Default: False (Bradley-Terry)")
    parser.add_argument("--value_head_dim", type=int, default=6,
                       help="Dimension of GPM value head (must be even)")
    parser.add_argument("--general_preference_tau", type=float, default=0.1,
                       help="Temperature for preference loss")
    parser.add_argument("--add_prompt_head", action="store_true", default=False)

    # Data arguments
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to grouped training dataset")
    parser.add_argument("--eval_dataset", type=str, default=None,
                       help="Path to grouped eval dataset (optional)")
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_eval_samples", type=int, default=None,
                       help="Limit eval samples (default: no limit)")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--max_comparisons_per_entry", type=int, default=None,
                       help="Limit comparisons per entry (None = no limit)")
    parser.add_argument("--train_split_ratio", type=float, default=1.0,
                       help="Ratio for train/val split if no eval_dataset provided")

    # Training arguments
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--micro_train_batch_size", type=int, default=1,
                       help="Batch size at entry level (1 is typical for grouped)")
    parser.add_argument("--accumulated_gradient", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                       choices=["cosine", "constant", "linear"])
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--margin_loss", action="store_true", default=False,
                       help="Use margin in loss (False = binary preference)")
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    # Checkpointing
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_best_model", type=int, default=3,
                       help="Keep top N models with lowest eval loss")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed arguments
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--zpg", type=int, default=1)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)

    # LoRA arguments
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")

    # Tokenizer
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # WandB
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="GPM-Grouped")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="gpm_grouped_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()

    # Validate args
    if args.is_general_preference:
        assert args.value_head_dim % 2 == 0, "value_head_dim must be even for GPM"

    train(args)
