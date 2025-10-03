# CS6886 Assignment 3

B Santhanakrishnan DA24S001

## Environment

-   Python 3.10 (recommended)
-   PyTorch 2.8.0 (CUDA 12.8 build)
-   wandb 0.22.1
-   opencv-python 4.12.0.88
-   tqdm 4.67.1
-   numpy 2.2.6

Random seeds are fixed in all scripts. CUDA determinism is not forced, so results may vary slightly across devices.

## Installation

This repo includes a `pyproject.toml`. If you are using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

This will create and sync a virtual environment with the declared dependencies.

If you prefer to use pip, you can create a virtual environment, and then:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train baseline MobileNetV2 on CIFAR-10

```bash
python main.py
```

This trains the model on CIFAR-10, logs results to `training_log.csv`, and saves the best model weights to `model.pth`.

If you want to skip this step (takes around 45m to 1.5hr depending on hardware), and directly run compression, then you can just run the below

```bash
cp saved_model.bin model.pth
```

and directly proceed to step 3.

### 2. Log training curves to Weights & Biases

```bash
python log_training_curves.py
```

Parses `training_log.csv` and logs loss/accuracy curves to wandb.

### 3. Compression (pruning + quantization aware training)

Example run:

```bash
python cs.py --prune_frac 0.3 --w_bits 4 --a_bits 8
```

Key arguments:

-   `--prune_frac`: fraction of expansion channels to prune
-   `--w_bits`: weight quantization bit-width
-   `--a_bits`: activation quantization bit-width

### 4. Failure analysis

```bash
python failures.py
```

Reloads `model.pth`, evaluates on CIFAR-10 test set, and logs a confusion matrix with counts and sample misclassified images to wandb.

## Reproducibility

-   Random seeds are fixed inside the scripts.
-   Results are deterministic given the seed and environment versions.
-   All wandb runs include full configuration information.

## Repository Link

Source code: https://github.com/porridgewithraisins/cs6886-a3

