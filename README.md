# DLP Assignment 02 — Augmentation & Robustness on Small-Data CIFAR-10


## Objective
Measure how data augmentation affects a compact Inception-style CNN trained on a very small CIFAR-10 subset, both on the clean test set and on a locally generated **shifted** test set.

## Setup
- **Data:** 500 images (50 per class) sampled with seed 42 → 400 train / 100 validation (stratified). Evaluation on the full 10,000-image clean CIFAR-10 test set.
- **Shifted benchmark:** the same 10,000 test images with random rotation (±25°), brightness/contrast changes, Gaussian blur (p=0.5), Gaussian noise (σ=12, p=0.5) and R/B channel swap (p=0.2). Labels verified identical to the clean set.
- **Model:** Conv-BN-ReLU stem + Inception-style module (1×1, 1×1→3×3, pooling branches, etc.), softmax over 10 classes (logits + sparse categorical cross-entropy).
- **Training:** Adam, batch size 32, up to 60 epochs, `ReduceLROnPlateau` (factor 0.5, patience 4), best checkpoint by validation loss. Seeds: **42, 123, 2026**.
- **Variants:** `none`, `geometric` (flip, rotation, translation), `photometric` (brightness, contrast), `combined`, `combined_cutout` (8×8 Cutout).

## Results

### Baseline (no augmentation, initial run)
Training was unstable at the start (validation loss in the thousands in epoch 1, e.g. 3816 for seed 42; validation accuracy near chance, 10%). After the loss settled, final train accuracy reached ~0.45–0.57 against validation ~0.26–0.38 — a clear generalisation gap with only 400 training images.

### Validation (mean of 3 seeds, at best-val-loss epoch)

| Variant | Mean Val Acc | Mean Val Loss |
|---|---|---|
| none | **0.2967** | **1.7537** |
| photometric | 0.2700 | 1.8294 |
| combined_cutout | 0.2700 | 1.8498 |
| geometric | 0.2533 | 1.8174 |
| combined | 0.2400 | 1.8801 |

Selected for final analysis: **`none`, seed 123** (lowest best validation loss, 1.6338).

### Test (mean ± std over 3 seeds)

| Variant | Clean Acc | Shifted Acc |
|---|---|---|
| none | **0.2905 ± 0.0123** | 0.2171 ± 0.0146 |
| geometric | 0.2753 ± 0.0127 | 0.2217 ± 0.0097 |
| photometric | 0.2737 ± 0.0351 | 0.2215 ± 0.0220 |
| combined | 0.2721 ± 0.0228 | **0.2221 ± 0.0264** |
| combined_cutout | 0.2617 ± 0.0312 | 0.2152 ± 0.0291 |

## Key Findings
1. **Augmentation did not improve absolute accuracy.** `none` had the best clean accuracy (29.1%) and the best validation scores. All models are well above chance (10%) but low overall, as expected with 400 training images.
2. **Augmentation reduced the clean→shifted drop.** The unaugmented model lost ~7.3 points under shift; augmented variants lost ~4.7–5.4 points, retaining ~80–82% of clean accuracy versus ~75%.
3. **Shifted accuracy is essentially tied** across variants (0.215–0.222). Differences are smaller than the seed-to-seed standard deviations, so no variant can be called a clear winner on robustness.
4. **Cutout and heavy combined augmentation added variance** (std up to ~0.03) without a gain, likely because the small model and dataset are already capacity/data-limited and the augmented training signal is harder to fit within 60 epochs.

## Limitations
- Only 3 seeds and a 100-image validation set make model selection noisy; selecting `none` over the others rests on small margins.
- The shifted benchmark is a local synthetic shift, not an external distribution shift dataset.
- Longer training or a gentler augmentation strength might let augmentation pay off; the augmented runs were still improving slowly at epoch 60.
- The baseline run used a high learning rate, which caused early instability.

## Conclusion
In this low-data regime, augmentation trades a little clean accuracy for a smaller drop under distribution shift, but it does not raise absolute accuracy and the shifted-set differences are within seed noise. More training data or longer training is likely the bigger lever than augmentation alone.