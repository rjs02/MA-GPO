# GPU Scaling for Qwen3-4B Training on hh-rlhf

## Overview

Training Qwen3-4B with max_len=2048 requires significant GPU memory. With **DeepSpeed ZeRO-3**, we can shard model parameters, gradients, and optimizer states across multiple GPUs to fit the model in memory while maintaining high effective batch sizes.

## Available Configurations

### Configuration 1: Batch Size 128 (Recommended)
**Script**: `train_pm_hh_rlhf.sh`

```bash
GPUs: 8x RTX 4090
Micro batch size per GPU: 2
Gradient accumulation: 8
Effective batch size: 2 × 8 × 8 = 128
Max sequence length: 2048
DeepSpeed: ZeRO-3
```

**Usage**:
```bash
sbatch scripts/rmpm/train_pm_hh_rlhf.sh
```

**Memory per GPU**: ~12-15GB (well within 24GB limit)

### Configuration 2: Batch Size 256 (High Throughput)
**Script**: `train_pm_hh_rlhf_bs256.sh`

```bash
GPUs: 16x RTX 4090
Micro batch size per GPU: 2
Gradient accumulation: 8
Effective batch size: 2 × 8 × 16 = 256
Max sequence length: 2048
DeepSpeed: ZeRO-3
```

**Usage**:
```bash
sbatch scripts/rmpm/train_pm_hh_rlhf_bs256.sh
```

**Memory per GPU**: ~8-10GB (very safe)

## How DeepSpeed ZeRO-3 Helps

DeepSpeed ZeRO-3 shards across GPUs:
1. **Model parameters** (4B params ≈ 16GB in bf16)
2. **Gradients** (another 16GB)
3. **Optimizer states** (32GB for AdamW with bf16)

**Total memory without ZeRO**: ~64GB (doesn't fit on 2x 4090s!)

**With ZeRO-3 on 8 GPUs**: ~8GB per GPU + activations

This allows us to train with:
- ✅ Full 2048 sequence length
- ✅ Large effective batch sizes (128-256)
- ✅ Gradient checkpointing for additional memory savings
- ✅ Flash Attention for efficient attention computation

## Scaling Formula

To calculate configurations for different effective batch sizes:

```
Effective Batch Size = micro_batch_size × gradient_accumulation × num_gpus
```

**Examples**:

| Target Batch Size | GPUs | Micro Batch | Grad Accum | Memory/GPU |
|-------------------|------|-------------|------------|------------|
| 64                | 4    | 2           | 8          | ~18GB      |
| 128               | 8    | 2           | 8          | ~12GB      |
| 256               | 16   | 2           | 8          | ~8GB       |
| 256               | 8    | 4           | 8          | ~18GB      |
| 512               | 16   | 4           | 8          | ~14GB      |

## Recommendations

1. **Start with batch size 128** (8 GPUs) - Good balance of speed and resource usage
2. **Use batch size 256** (16 GPUs) if:
   - You want maximum throughput
   - You have many jobs to run in parallel
   - You want very stable training

3. **Don't go below 4 GPUs** for Qwen3-4B with max_len=2048 - memory will be tight

## Performance Notes

- **Gradient accumulation** doesn't add memory overhead, but increases training time
- **More GPUs = faster training** (linear speedup with good communication)
- **ZeRO-3** adds some communication overhead but enables much larger models
- **Gradient checkpointing** reduces memory by ~30-40% at cost of ~20% slower training

## Troubleshooting

### Still getting OOM?

1. **Reduce micro_batch_size**: Try 1 per GPU (increase grad_accum to compensate)
2. **Add more GPUs**: Memory per GPU decreases linearly with more GPUs
3. **Reduce max_len**: Try 1536 or 1024 if absolutely necessary

### Training too slow?

1. **Increase micro_batch_size**: If you have memory headroom, try 3-4 per GPU
2. **Reduce gradient_accumulation**: Decrease to 4 (but need more GPUs for same effective batch size)
3. **Check GPU utilization**: Use `nvidia-smi dmon` to ensure GPUs are busy

### Communication bottleneck?

With many GPUs (16+), inter-GPU communication can slow things down:
1. Ensure all GPUs are on the same node if possible
2. Use InfiniBand if available
3. Consider ZeRO-2 instead of ZeRO-3 if you have enough memory

## Dataset Size

- **Training samples**: 160,791
- **Test samples**: 8,550
- **Steps per epoch** (batch size 128): ~1,256 steps
- **Steps per epoch** (batch size 256): ~628 steps

## Estimated Training Time

With 8x RTX 4090:
- **Batch size 128**: ~4-5 hours for 1 epoch
- **Evaluation**: ~30 seconds every 100 steps

With 16x RTX 4090:
- **Batch size 256**: ~2-3 hours for 1 epoch
- **Evaluation**: ~30 seconds every 100 steps

*(Times are approximate and depend on actual GPU availability and cluster load)*
