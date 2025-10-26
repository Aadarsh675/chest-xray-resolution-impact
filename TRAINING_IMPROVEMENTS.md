# Training Improvements Made

## Issues Fixed

### 1. WandB Logging Issues
**Problem**: Metrics not logging properly every batch iteration

**Solution**:
- Added proper step tracking with `global_step` variable
- Changed metric names to be more descriptive (`train/batch_loss` instead of `train/loss`)
- Added learning rate logging every batch
- Implemented step-based logging with `step=current_step` parameter

### 2. Training Instability
**Problem**: Loss fluctuating wildly (e.g., 0.0857 → 5.7102)

**Root Causes Identified**:
1. Zero-padding in collate function creating artificial edges
2. No gradient clipping (common in transformer models)
3. Learning rate too high (1e-4)
4. No learning rate scheduling
5. No regularization (weight decay)

**Solutions Applied**:

#### a. Improved Padding Strategy
- Changed from zero-padding to ImageNet mean padding
- Better maintains statistical properties of images
- Ensures dimensions are multiples of 32 for GPU efficiency

#### b. Gradient Clipping
- Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Prevents exploding gradients common in transformer models

#### c. Learning Rate Reduction
- Reduced from `1e-4` to `5e-5`
- More conservative for object detection tasks

#### d. Learning Rate Scheduler
- Added Cosine Annealing scheduler
- Gradually decreases LR over epochs for better convergence
- Starts at `5e-5`, ends at `1e-6`

#### e. Weight Decay
- Added `weight_decay=1e-4` to AdamW optimizer
- Provides regularization to prevent overfitting

## Code Changes Summary

### Modified Files
- `src/train.py`

### Key Changes

1. **train_one_epoch function**:
   - Added global step tracking
   - Improved wandb logging with proper step indexing
   - Added learning rate logging
   - Added gradient clipping

2. **collate_fn function**:
   - Changed from zero-padding to mean-value padding
   - Added dimension rounding to multiples of 32
   - Better documentation

3. **train_and_validate function**:
   - Added weight decay to optimizer
   - Added cosine annealing scheduler
   - Added scheduler step at end of each epoch
   - Added learning rate logging

4. **Learning rate**:
   - Reduced from `1e-4` to `5e-5`

## Expected Improvements

### Stability
- More stable training with reduced loss fluctuation
- Gradual convergence instead of wild swings
- Better handling of difficult batches

### Performance
- Better final mAP scores
- More consistent validation metrics
- Improved generalization

### Monitoring
- Better wandb tracking with step-based logging
- Learning rate visualization
- Per-batch metrics for debugging

## New WandB Logs

The following metrics are now logged to WandB:

### Every Batch
- `train/batch_loss`: Loss value for each batch
- `train/epoch`: Current epoch number
- `train/learning_rate`: Current learning rate

### Every Epoch
- `train/epoch_avg_loss`: Average loss across epoch
- `learning_rate`: Scheduler-adjusted learning rate
- All existing validation and test metrics

## Recommendations

1. **Monitor the first epoch carefully**: The improvements should show more stable loss within the first epoch
2. **Watch learning rate decay**: Check that LR decreases smoothly over epochs
3. **Compare validation metrics**: Improvement should be visible in mAP scores
4. **Adjust batch size if needed**: With these changes, you might be able to use a slightly larger batch size

## Next Steps

After these improvements, you should see:
- Smoother loss curves in WandB
- Gradual decrease in loss over epochs
- More stable validation metrics
- Better overall detection performance
