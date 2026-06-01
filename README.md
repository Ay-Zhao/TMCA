# TMCA

TMCA is a Python research code repository for multimodal molecular property prediction. The code combines 1D molecular strings, 2D graph features, and 3D Uni-Mol-style molecular representations for binary and multi-label classification experiments.

This repository is intended to support the paper link and provide the source code needed to reproduce or adapt the training workflow.

## Repository structure

```text
src/
  training/
    train_GGT_2.py              # binary/single-task training entry point
    train_GGT_multilabel.py     # multi-label training entry point
    GGTmodel_2.py               # multimodal model definition
    CustomizedDataset_GGT_InM.py
    util.py
  branch_3D/
    model_unimol.py             # 3D molecular representation branch
    MoleculeDataset.py
    util.py
    config/default.yaml
    uni_tool/                   # Uni-Mol utility modules and vocabulary
```

## Environment

The code requires Python and common scientific machine-learning packages, including:

- PyTorch
- PyTorch Geometric
- torch-scatter
- torch-sparse
- transformers
- RDKit
- scikit-learn
- pandas
- numpy
- scipy
- tqdm
- matplotlib
- Pillow
- addict
- PyYAML
- TensorBoard

Install package versions compatible with your CUDA/PyTorch environment. For PyTorch Geometric packages, follow the official installation command for your local PyTorch and CUDA versions.

## Data

Training scripts expect datasets under:

```text
src/dataset_GGT/
```

The default dataset name is `sider`. You can select another dataset with `--dataset`.

Generated training outputs are written to folders such as `results/`, `results_bestValidation/`, and `log/`.

## Usage

Run commands from the repository root.

Single-task or binary classification:

```bash
python -m src.training.train_GGT_2 --dataset bbbp --n_tasks 1 --epochs 100 --batch_size 16
```



Useful options include:

- `--enabled_modality 1d 2d 3d`
- `--fuse_mechanism plus`
- `--learning_rate 1e-3`
- `--weight_decay 4e-4`
- `--random_seed 0`

## Notes

Large checkpoints, datasets, logs, and result files are not tracked in Git. Add the required datasets and model weights locally before running experiments.
