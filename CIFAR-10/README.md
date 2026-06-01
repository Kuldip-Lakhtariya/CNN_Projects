# CIFAR-10 Image Classification

Custom CNN trained from scratch achieving **80.6% validation accuracy** on 10-class image classification.

## Architecture
Input (32×32×3)
→ Conv(32) → BatchNorm → MaxPool
→ Conv(64) → BatchNorm → MaxPool
→ Conv(128) → BatchNorm → MaxPool
→ Dropout(0.25)
→ Dense(128) → Softmax(10)

## Results
| Metric | Value |
|--------|-------|
| Validation Accuracy | 80.6% |
| Parameters | ~1.2M |
| Epochs | 50 |

## Key Learnings
- BatchNorm before activation stabilizes training
- GAP vs Flatten — GAP reduces parameters and overfitting risk
- Transfer learning on upscaled CIFAR-10 failed — EfficientNet filters 
  expect natural image statistics, not pixelated 32×32 upscales
- Diagnosed underfitting vs overfitting from loss curves alone

## Stack
Python · TensorFlow/Keras · Jupyter
