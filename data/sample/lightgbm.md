
# LightGBM Architecture

LightGBM is a gradient boosting framework.

It uses two important techniques:

1. Gradient-based One-Side Sampling (GOSS)
2. Exclusive Feature Bundling (EFB)

GOSS reduces the number of data instances used during training.

EFB reduces the number of features by bundling mutually exclusive features.

LightGBM is designed for efficient training on large datasets.
