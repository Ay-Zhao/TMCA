import argparse
import os
from datetime import datetime
from io import BytesIO
import re
import pandas as pd
# from torch_geometric.loader import DataLoader
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from itertools import compress
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import pandas as pds

import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from math import pi as PI
from torch.utils.data import Subset, DataLoader
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset_GGT"


def _safe_name(value) -> str:
    """
    Convert a value into a filesystem-safe string for TensorBoard log_dir.
    Removes brackets, quotes, colons, spaces, and other unsafe characters.
    """
    if value is None:
        return "none"

    if isinstance(value, (list, tuple, set)):
        value = "_".join(str(v) for v in value)
    else:
        value = str(value)

    value = value.strip()
    value = value.replace("\\", "_").replace("/", "_")
    value = value.replace(":", "-")
    value = value.replace(" ", "")
    value = re.sub(r"[^A-Za-z0-9._=-]+", "_", value)

    return value


def def_log_dir(args) -> str:
    model_name = _safe_name(args.model_name)
    dataset = _safe_name(args.dataset)
    epochs = _safe_name(args.epochs)
    batch_size = _safe_name(args.batch_size)
    fuse_mechanism = _safe_name(args.fuse_mechanism)
    enabled_modality = _safe_name(args.enabled_modality)
    random_seed = _safe_name(args.random_seed)
    learning_rate = _safe_name(args.learning_rate)
    weight_decay = _safe_name(args.weight_decay)
    key = _safe_name(args.key)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_log_dir = Path("log") / (
        f"{model_name}_"
        f"{dataset}_"
        f"ep{epochs}_"
        f"bs{batch_size}_"
        f"fuse{fuse_mechanism}_"
        f"mod{enabled_modality}_"
        f"seed{random_seed}_"
        f"lr{learning_rate}_"
        f"wd{weight_decay}_"
        f"key{key}_"
        f"{timestamp}"
    )

    tensorboard_log_dir = str(tensorboard_log_dir)

    print(f"TensorBoard log_dir: {tensorboard_log_dir}")
    return tensorboard_log_dir


def model_eval(args, model, device, loader, tokenizer, criterion):
    eval_loss = 0
    accuracy = 0
    auc_roc = 0
    preds = []
    labels = []
    for i, data in enumerate(tqdm(loader)):
        with torch.no_grad():
            data = data.to(device)
            smiles = data.smiles
            if args.token_length_smile == 0:
                inputs = tokenizer.batch_encode_plus(smiles, truncation=True, padding=True, return_tensors="pt")
            else:
                inputs = tokenizer.batch_encode_plus(smiles, max_length=args.token_length_smile,
                                                     truncation=True, pad_to_max_length=True, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(data, inputs)
            data.y = data.y.view(len(data), args.n_tasks)
            # data.y = data.y[:, 0].view(-1, 1)
            eval_loss += criterion(outputs, data.y)

            preds += outputs.view(-1).cpu().tolist()
            labels += data.y.cpu().tolist()

    trues = np.array(labels).reshape(-1, args.n_tasks).T
    belief_scores = np.array(preds).reshape(-1, args.n_tasks).T
    roc_auc_score_list = []
    for i in range(args.n_tasks):
        temp_roc_auc_score = roc_auc_score(trues[i].tolist(), belief_scores[i].tolist())
        roc_auc_score_list.append(temp_roc_auc_score)

    auc_roc = sum(roc_auc_score_list) / args.n_tasks
    accuracy = accuracy_score(labels, [1 if value > 0.5 else 0 for value in preds])

    return eval_loss, accuracy, auc_roc


def plot_confusion_matrix_image(y_pred, y_true, class_labels=["Class 0", "Class 1"]):
    print(y_true)
    print(y_pred)
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    disp.plot(cmap="Blues", ax=ax, values_format=".0f")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Convert the BytesIO object to a PIL Image and then to a torch tensor
    image = Image.open(buf)
    image = torch.tensor(np.array(image))
    return image

def save_model(best_epoch, model_state_dict, optimizer_state_dict, best_loss, log_directory):
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': best_loss,
    }, os.path.join(log_directory, 'checkpoint.pth'))


def split_data(our_dataset, random_seed, train_batch_size, is_balance=False):
    print("size of dataset", len(our_dataset))

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(random_seed)

    train_size = int(0.8 * len(our_dataset))
    valid_size = int(0.1 * len(our_dataset))
    test_size = len(our_dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(our_dataset, [train_size, valid_size, test_size])

    #  balanced training
    if is_balance:
        n_y_positive = 0
        n_y_negative = 0
        idx_balanced_train_dataset = []

        for idx in train_dataset.indices:
            if our_dataset[idx].y == 1:
                n_y_positive += 1
                idx_balanced_train_dataset.append(idx)

        for idx in train_dataset.indices:
            if our_dataset[idx].y == 0:
                n_y_negative += 1
                idx_balanced_train_dataset.append(idx)
                if n_y_negative == n_y_positive:
                    break

        balanced_train_dataset = torch.utils.data.Subset(our_dataset, idx_balanced_train_dataset)
        print("the number of balanced training dataset ", len(balanced_train_dataset))
        train_dataset = balanced_train_dataset

    """ data loader """
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=train_batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size,shuffle=True)

    return train_loader, valid_loader, test_loader, train_size, valid_size, test_size



def add_regression_data_number_scaler(writer, number):
    writer.add_scalar('Number of testing data', number)

def compute_std_mean(result):
    nums = np.array(result)
    std = np.std(np.array(nums))
    mean = np.mean(nums)
    print("reuslt: {} | {:.4} + {:.3}".format(nums, mean, std))



def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def random_scaffold_split(dataset, task_idx=None, null_value=0, smiles_list=[],
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0, batch_size=16, collate_fn=None,
                          save_split=False, split_name=None, split_dir=None):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.RandomState(seed)

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    # if task_idx != None:
    #     # filter based on null values in task_idx
    #     # get task array
    #     y_task = np.array([data.y[task_idx].item() for data in dataset])
    #     # boolean array that correspond to non null values
    #     non_null = y_task != null_value
    #     smiles_list = list(compress(enumerate(smiles_list), non_null))
    # else:
    #     non_null = np.ones(len(dataset)) == 1
    #     smiles_list = list(compress(enumerate(smiles_list), non_null))

    if smiles_list == []:
        for tmp in dataset:
            if tmp is not None:
                smiles_list.append(tmp.smiles)
    non_null = np.ones(len(dataset)) == 1
    smiles_list = list(compress(enumerate(smiles_list), non_null))


    scaffolds = defaultdict(list)
    invalid_smiles = []
    for ind, smiles in smiles_list:
        try:
            scaffold = generate_scaffold(smiles, include_chirality=True)
            scaffolds[scaffold].append(ind)
        except ValueError as e:
            invalid_smiles.append(smiles)
    pd.DataFrame({'invalid smiles': invalid_smiles}).to_csv("{}_without_scaffold".format('dataset'), index=False)



    # random
    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))
    # scaffold_sets = np.array(list(scaffolds.values()), dtype=object)

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    ''' new add on for single branch of 3D'''
    train_idx = [int(i) for i in train_idx]
    valid_idx = [int(i) for i in valid_idx]
    test_idx = [int(i) for i in test_idx]

    # save split here
    if save_split:
        if split_dir is None:
            split_dir = Path("split") / (split_name if split_name else "default_split")
        else:
            split_dir = Path(split_dir)

        split_dir.mkdir(parents=True, exist_ok=True)

        np.save(split_dir / "train_idx.npy", np.array(train_idx, dtype=np.int64))
        np.save(split_dir / "valid_idx.npy", np.array(valid_idx, dtype=np.int64))
        np.save(split_dir / "test_idx.npy", np.array(test_idx, dtype=np.int64))

        with open(split_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "seed": seed,
                "frac_train": frac_train,
                "frac_valid": frac_valid,
                "frac_test": frac_test,
                "dataset_size": len(dataset)
            }, f, indent=2)

        with open(split_dir / "smiles.csv", "w", encoding="utf-8") as f:
            f.write("idx,smiles\n")
            for i, s in enumerate(smiles_list):
                f.write(f"{i},\"{s}\"\n")


    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    ''' mark 
    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    '''
    # tmp = next(iter(train_loader))
    # print(tmp[10])
    # tmp = next(iter(val_loader))
    # print(tmp[10])
    # tmp = next(iter(test_loader))
    # print(tmp[10])

    return train_loader, valid_loader, test_loader, len(train_idx), len(valid_idx), len(test_idx)


# =============================================================================
# Model config normalization
# =============================================================================

VALID_MODALITIES = {"1d", "2d", "3d"}
VALID_FUSIONS = {"single", "crossattn", "concat", "plus"}


def normalize_modalities(enabled_modality):
    """
    Convert args.enabled_modality into a tuple accepted by Net.

    Supported input examples:
        "1d"
        "2d"
        "3d"
        "1d 2d"
        "1d,2d"
        "1d_2d"
        "1d_2d_3d"
        ["1d", "2d"]
        ("1d", "2d", "3d")

    Output:
        ("1d",)
        ("1d", "2d")
        ("1d", "2d", "3d")
    """
    if enabled_modality is None:
        raise ValueError("args.enabled_modality is None.")

    if isinstance(enabled_modality, str):
        s = enabled_modality.strip()

        # Handle common forms: "1d 2d", "1d,2d", "1d_2d_3d"
        s = s.replace(",", " ")
        parts = s.split()

        # If the string is like "1d_2d_3d", split by "_"
        if len(parts) == 1 and "_" in parts[0]:
            parts = parts[0].split("_")

        modalities = tuple(parts)

    elif isinstance(enabled_modality, (list, tuple)):
        modalities = tuple(enabled_modality)

    else:
        raise TypeError(
            f"Unsupported enabled_modality type: {type(enabled_modality)}. "
            "Expected str, list, or tuple."
        )

    # Remove empty strings
    modalities = tuple(m for m in modalities if m != "")

    unknown = set(modalities) - VALID_MODALITIES
    if unknown:
        raise ValueError(
            f"Unknown modalities: {unknown}. "
            f"Valid modalities are: {VALID_MODALITIES}"
        )

    if len(modalities) == 0:
        raise ValueError("enabled_modality cannot be empty.")

    # Preserve the input order, but remove duplicates
    deduped = []
    for m in modalities:
        if m not in deduped:
            deduped.append(m)

    modalities = tuple(deduped)

    if len(modalities) > 3:
        raise ValueError(f"Too many modalities: {modalities}")

    return modalities


def normalize_fusion(fuse_mechanism, modalities):
    """
    Convert args.fuse_mechanism into the fusion name accepted by Net.

    Model Net accepts:
        "single"
        "crossattn"
        "concat"
        "plus"

    This function also supports common aliases:
        "cross_attention" -> "crossattn"
        "cross_attn"      -> "crossattn"
        "concatenate"     -> "concat"
        "add"             -> "plus"
    """
    if fuse_mechanism is None:
        raise ValueError("args.fuse_mechanism is None.")

    fusion = str(fuse_mechanism).strip().lower()

    alias_map = {
        "single": "single",

        "crossattn": "crossattn",
        "cross_attention": "crossattn",
        "cross-attention": "crossattn",
        "cross_attn": "crossattn",

        "concat": "concat",
        "concatenate": "concat",
        "concatenation": "concat",

        "plus": "plus",
        "add": "plus",
        "sum": "plus",
    }

    if fusion not in alias_map:
        raise ValueError(
            f"Unknown fuse_mechanism: {fuse_mechanism}. "
            f"Valid fusion values are: {VALID_FUSIONS}"
        )

    fusion = alias_map[fusion]

    # Single modality must use fusion="single"
    if len(modalities) == 1:
        if fusion != "single":
            print(
                f"[Config Warning] Single modality {modalities} was given "
                f"fusion='{fusion}'. Automatically changing fusion to 'single'."
            )
        fusion = "single"

    # Multi-modality cannot use single fusion
    if len(modalities) > 1 and fusion == "single":
        raise ValueError(
            f"fusion='single' only supports one modality, "
            f"but got modalities={modalities}."
        )

    return fusion


