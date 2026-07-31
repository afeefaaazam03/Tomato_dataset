# Configs

`tomato_benchmark.yaml` is the single source of truth for: dataset paths, class list, train/val/test
split ratios, preprocessing/augmentation parameters, per-task defaults, and the model zoo (which
`timm`/`torchvision` architectures are available per task, and their hyperparameter overrides).

Every script under `scripts/` takes `--config path/to/this.yaml`; copy this file if you want an
alternate configuration (e.g. a different crop padding or a reduced model zoo for CI smoke tests).
