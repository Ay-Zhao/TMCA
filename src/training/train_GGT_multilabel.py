import os
import time
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from src.training import util as sd
from src.training.util import DATASET_DIR
from src.branch_3D import UniMolModel
from src.training.CustomizedDataset_GGT_InM import MyOwnDataset
from src.training.GGTmodel_2 import Net


# =============================================================================
# Result directories / Slurm info
# =============================================================================

os.makedirs("results", exist_ok=True)

job_id = os.environ.get("SLURM_JOB_ID", "nojid")
task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "notask")


# =============================================================================
# Model builder
# =============================================================================

def build_model_from_args(args, number_of_task, device):
    """
    Build Net using normalized modality and fusion config.

    This is the only place where args.enabled_modality and args.fuse_mechanism
    should be converted before passing into Net.
    """
    modalities = sd.normalize_modalities(args.enabled_modality)
    fusion = sd.normalize_fusion(args.fuse_mechanism, modalities)

    # Keep args synchronized, so logging / TensorBoard / checkpoint names
    # reflect the actual model config.
    args.enabled_modality = modalities
    args.fuse_mechanism = fusion

    print("========== Model Config ==========")
    print("enabled_modalities:", modalities)
    print("fusion:", fusion)
    print("number_of_task:", number_of_task)
    print("==================================")

    model = Net(
        n_output_layers=number_of_task,
        fusion=fusion,
    ).to(device)

    return model


# =============================================================================
# Multi-label helpers
# =============================================================================

def prepare_multilabel_target_and_mask(args, data, graph, outputs, device):
    """
    Prepare target and mask for multi-label or separate-label classification.

    Normal SIDER multi-label:
        outputs: [B, 27]
        y:       [B, 27]
        mask:    [B, 27]

    SIDER separate-label:
        outputs: [B, 1]
        y:       [B, 1]
        mask:    [B, 1]
    """
    y = torch.as_tensor(
        data["target"],
        device=device,
        dtype=torch.float32,
    )

    # First prepare original full-size mask before selecting one task.
    if hasattr(graph, "y_mask"):
        y_mask = graph.y_mask.to(device).bool()
        if y_mask.shape != y.shape:
            y_mask = y_mask.view_as(y)
    else:
        y_mask = ~torch.isnan(y)

    y_mask = y_mask & (~torch.isnan(y))

    # Select one SIDER label if using separate-label mode.
    y, y_mask = select_target_and_mask_if_needed(args, y, y_mask)

    # Now y should match outputs.
    if y.shape != outputs.shape:
        y = y.view_as(outputs)

    if y_mask.shape != outputs.shape:
        y_mask = y_mask.view_as(outputs)

    # BCE cannot consume NaN labels.
    y = torch.nan_to_num(y, nan=0.0)

    return y, y_mask


def masked_bce_loss(outputs, y, y_mask, criterion):
    """
    Compute masked BCEWithLogitsLoss.

    criterion should be:
        torch.nn.BCEWithLogitsLoss(reduction="none")
    """
    loss_raw = criterion(outputs.float(), y.float())

    if y_mask.sum() == 0:
        # This should almost never happen, but prevents crashing.
        return loss_raw.mean() * 0.0

    loss = loss_raw[y_mask].mean()
    return loss


def multilabel_metrics(labels, preds, masks=None, threshold=0.5):
    """
    Compute task-wise average AUC and task-wise average accuracy.

    Args:
        labels: [N, T]
        preds:  [N, T], sigmoid probabilities
        masks:  [N, T], bool, True means valid label

    Returns:
        accuracy: task-wise mean accuracy
        auc:      task-wise mean ROC-AUC
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    if masks is None:
        masks = np.ones_like(labels, dtype=bool)
    else:
        masks = np.asarray(masks).astype(bool)

    n_tasks = labels.shape[1]

    auc_list = []
    acc_list = []

    for task_idx in range(n_tasks):
        valid = masks[:, task_idx]

        if valid.sum() == 0:
            continue

        y_true = labels[valid, task_idx]
        y_score = preds[valid, task_idx]

        # Accuracy is still defined even if only one class exists.
        y_pred = (y_score > threshold).astype(int)
        acc_list.append(accuracy_score(y_true, y_pred))

        # AUC is undefined if only one class appears in this task split.
        if len(np.unique(y_true)) >= 2:
            auc_list.append(roc_auc_score(y_true, y_score))
        else:
            print(
                f"[Metric warning] task {task_idx} has only one class in this split; "
                f"skip AUC for this task."
            )

    accuracy = float(np.mean(acc_list)) if len(acc_list) > 0 else float("nan")
    auc = float(np.mean(auc_list)) if len(auc_list) > 0 else float("nan")

    return accuracy, auc


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def model_eval(args, model, device, loader, num_sample, tokenizer, criterion):
    """
    Evaluate model on a data loader for multi-label classification.
    """
    model.eval()

    eval_loss_sum = 0.0
    preds_list = []
    labels_list = []
    masks_list = []

    use_tqdm = sys.stdout.isatty()
    for data in tqdm(loader, disable=not use_tqdm):
        smiles = data["graph"].smiles

        if args.token_length_smile == 0:
            inputs = tokenizer.batch_encode_plus(
                smiles,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = tokenizer.batch_encode_plus(
                smiles,
                max_length=args.token_length_smile,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        inputs = inputs.to(device)
        graph = data["graph"].to(device)
        unimol_input = {k: v.to(device) for k, v in data["unimol_input"].items()}

        outputs = model(
            graph=graph,
            inputs=inputs,
            unimol_input=unimol_input,
        )

        y, y_mask = prepare_multilabel_target_and_mask(
            args=args,
            data=data,
            graph=graph,
            outputs=outputs,
            device=device,
        )

        loss = masked_bce_loss(
            outputs=outputs,
            y=y,
            y_mask=y_mask,
            criterion=criterion,
        )

        batch_size = y.size(0)
        eval_loss_sum += loss.item() * batch_size

        batch_preds = torch.sigmoid(outputs).detach().cpu().numpy()
        batch_labels = y.detach().cpu().numpy()
        batch_masks = y_mask.detach().cpu().numpy().astype(bool)

        preds_list.append(batch_preds)
        labels_list.append(batch_labels)
        masks_list.append(batch_masks)

    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)

    accuracy, auc_roc = multilabel_metrics(
        labels=labels,
        preds=preds,
        masks=masks,
    )

    eval_loss = eval_loss_sum / num_sample

    return eval_loss, accuracy, auc_roc


# =============================================================================
# Optimizer / Scheduler factory
# =============================================================================

def build_optimizer_and_scheduler(model, args, T_max):
    """
    AdamW + CosineAnnealingLR.

    Only parameters with requires_grad=True are passed into optimizer.
    This is important when using model.set_train_stage("fusion_only").
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, T_max),
    )

    return optimizer, scheduler

def select_target_and_mask_if_needed(args, y, y_mask):
    """
    For SIDER separate-label training, select one label column.

    Original:
        y:      [B, 27]
        y_mask: [B, 27]

    New:
        y:      [B, 1]
        y_mask: [B, 1]
    """
    if args.dataset.lower() == "sider" and getattr(args, "target_task", -1) >= 0:
        if y.ndim == 1:
            raise ValueError(
                "Expected SIDER target to have shape [batch_size, 27], "
                f"but got shape {tuple(y.shape)}"
            )

        y = y[:, args.target_task].unsqueeze(1)
        y_mask = y_mask[:, args.target_task].unsqueeze(1)

    return y, y_mask

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = sd.parse_input()

    sd.set_global_seed(args.random_seed)

    epochs = args.epochs
    number_of_task = args.n_tasks

    # For SIDER separate-label training:
    # original SIDER has 27 labels, but each run predicts only one label.
    epochs = args.epochs
    fusion_end_epochs = args.freeze_epoch

    # Default: normal training
    number_of_task = args.n_tasks

    # SIDER separate-label training mode:
    # original SIDER has 27 labels, but each run predicts only one label.
    if args.dataset.lower() == "sider" and getattr(args, "target_task", -1) >= 0:
        assert 0 <= args.target_task < 27, "For SIDER, target_task should be 0-26."

        args.n_tasks = 1
        number_of_task = 1

        print(
            f"[SIDER separate-label mode] "
            f"target_task={args.target_task}, model_output_dim=1"
        )

    # Normal SIDER multi-label mode
    elif args.dataset.lower() == "sider":
        assert number_of_task == 27, (
            f"Normal SIDER multi-label training should use --n_tasks 27, "
            f"but got --n_tasks {number_of_task}"
        )

    if number_of_task <= 1:
        print(
            "[Info] Single-label/binary mode. "
            f"Current n_tasks={number_of_task}."
        )

    # ── TensorBoard ──────────────────────────────────────────────────────────
    tensorboard_log_dir = "log_GGT/" + sd.def_log_dir(args)
    writer = SummaryWriter(tensorboard_log_dir)

    # ── Device & tokenizer ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # ── Dataset ──────────────────────────────────────────────────────────────
    molecular_dataset = MyOwnDataset(
        str(DATASET_DIR / args.dataset),
        dataset_name=args.dataset,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model_from_args(
        args=args,
        number_of_task=number_of_task,
        device=device,
    )

    # ── Initialize staged training ───────────────────────────────────────────
    if fusion_end_epochs > 0:
        current_stage = "fusion_only"
    else:
        current_stage = "unfreeze_all"

    model.set_train_stage(current_stage)

    print(f"[Train stage init] {current_stage}")

    model_batch = UniMolModel(
        output_dim=767,
        data_type="molecule",
        remove_hs=False,
    )

    # ── Data split ───────────────────────────────────────────────────────────
    is_balance = False

    if args.random_scaffold:
        print("---- in random scaffold")

        (
            train_loader,
            valid_loader,
            test_loader,
            train_size,
            valid_size,
            test_size,
        ) = sd.random_scaffold_split(
            dataset=molecular_dataset,
            null_value=0,
            smiles_list=[],
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.random_seed,
            batch_size=args.batch_size,
            collate_fn=model_batch.batch_collate_fn_2,
        )
    else:
        # sd.split_data returns only train / test splits.
        # valid is aliased to test.
        train_loader, test_loader, train_size, test_size = sd.split_data(
            molecular_dataset,
            args.random_seed,
            args.batch_size,
            is_balance,
        )

        valid_loader = test_loader
        valid_size = test_size

    if args.random_scaffold:
        total_data = train_size + valid_size + test_size
        print("The length of data:", total_data)
        print("train_size, valid_size, test_size:", train_size, valid_size, test_size)
    else:
        print("The length of data (train + test, valid=test):", train_size + test_size)
        print("train_size, test_size (valid=test):", train_size, test_size)

    print(
        "len(train_loader), len(valid_loader), len(test_loader):",
        len(train_loader),
        len(valid_loader),
        len(test_loader),
    )

    # ── Loss ─────────────────────────────────────────────────────────────────
    # Important: reduction="none" is needed for masked multi-label BCE.
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    # ── Optimizer / scheduler ────────────────────────────────────────────────
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        args=args,
        T_max=epochs,
    )

    # ── Metric history ───────────────────────────────────────────────────────
    train_loss_list = []
    train_accuracy_list = []
    train_auc_roc_list = []

    valid_loss_list = []
    valid_accuracy_list = []
    valid_auc_roc_list = []

    test_loss_list = []
    test_accuracy_list = []
    test_auc_roc_list = []

    # ── Track best model on disk ─────────────────────────────────────────────
    best_valid_auc = -1.0
    best_epoch = 0

    checkpoint_dir = "./checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_name = (
        f"{checkpoint_dir}/multilabel_3d_branch_"
        f"{args.dataset}_task{getattr(args, 'target_task', -1)}_"
        f"{args.model_name}.pth"
    )

    # =============================================================================
    # Training loop
    # =============================================================================

    for epoch in range(epochs):
        # ── Switch from fusion_only to unfreeze_all at freeze_epoch ───────────
        if epoch == fusion_end_epochs and fusion_end_epochs > 0:
            current_stage = "unfreeze_all"
            model.set_train_stage(current_stage)

            remaining = epochs - fusion_end_epochs

            optimizer, scheduler = build_optimizer_and_scheduler(
                model=model,
                args=args,
                T_max=remaining,
            )

            print(f"[Train stage switch] epoch={epoch}, stage={current_stage}")

        start_time = time.time()

        # model.train() sets all submodules to train mode,
        # so we re-apply set_train_stage() after it.
        model.train()
        model.set_train_stage(current_stage)

        print(f"[Epoch {epoch}] current_stage = {current_stage}")

        train_loss_sum = 0.0
        preds_list = []
        labels_list = []
        masks_list = []

        use_tqdm = sys.stdout.isatty()
        for data in tqdm(train_loader, disable=not use_tqdm):
            smiles = data["graph"].smiles

            if args.token_length_smile == 0:
                inputs = tokenizer.batch_encode_plus(
                    smiles,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = tokenizer.batch_encode_plus(
                    smiles,
                    max_length=args.token_length_smile,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

            inputs = inputs.to(device)
            graph = data["graph"].to(device)
            unimol_input = {k: v.to(device) for k, v in data["unimol_input"].items()}

            outputs = model(
                graph=graph,
                inputs=inputs,
                unimol_input=unimol_input,
            )

            y, y_mask = prepare_multilabel_target_and_mask(
                args=args,
                data=data,
                graph=graph,
                outputs=outputs,
                device=device,
            )

            loss = masked_bce_loss(
                outputs=outputs,
                y=y,
                y_mask=y_mask,
                criterion=criterion,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            train_loss_sum += loss.item() * batch_size

            batch_preds = torch.sigmoid(outputs).detach().cpu().numpy()
            batch_labels = y.detach().cpu().numpy()
            batch_masks = y_mask.detach().cpu().numpy().astype(bool)

            preds_list.append(batch_preds)
            labels_list.append(batch_labels)
            masks_list.append(batch_masks)

        scheduler.step()

        # ── Train metrics ────────────────────────────────────────────────────
        train_loss = train_loss_sum / train_size
        print("Train/Loss", train_loss, epoch)

        train_preds = np.concatenate(preds_list, axis=0)
        train_labels = np.concatenate(labels_list, axis=0)
        train_masks = np.concatenate(masks_list, axis=0)

        train_accuracy, train_auc_roc = multilabel_metrics(
            labels=train_labels,
            preds=train_preds,
            masks=train_masks,
        )

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Acc", train_accuracy, epoch)
        writer.add_scalar("Train/AUC-ROC", train_auc_roc, epoch)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        train_auc_roc_list.append(train_auc_roc)

        # ── Validation & test ────────────────────────────────────────────────
        valid_loss, valid_accuracy, valid_auc_roc = model_eval(
            args=args,
            model=model,
            device=device,
            loader=valid_loader,
            num_sample=valid_size,
            tokenizer=tokenizer,
            criterion=criterion,
        )

        test_loss, test_accuracy, test_auc_roc = model_eval(
            args=args,
            model=model,
            device=device,
            loader=test_loader,
            num_sample=test_size,
            tokenizer=tokenizer,
            criterion=criterion,
        )

        writer.add_scalar("Valid/Loss", valid_loss, epoch)
        writer.add_scalar("Valid/Acc", valid_accuracy, epoch)
        writer.add_scalar("Valid/AUC_ROC", valid_auc_roc, epoch)

        writer.add_scalar("Test/Loss", test_loss, epoch)
        writer.add_scalar("Test/Acc", test_accuracy, epoch)
        writer.add_scalar("Test/AUC_ROC", test_auc_roc, epoch)

        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)
        valid_auc_roc_list.append(valid_auc_roc)

        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        test_auc_roc_list.append(test_auc_roc)

        # ── Save best model by validation AUC ────────────────────────────────
        # If valid_auc_roc is NaN, do not update best model.
        # if not np.isnan(valid_auc_roc) and valid_auc_roc > best_valid_auc:
        #     best_valid_auc = valid_auc_roc
        #     best_epoch = epoch
        #
        #     torch.save(
        #         {
        #             "model_state": model.state_dict(),
        #             "optimizer_state": optimizer.state_dict(),
        #             "epoch": best_epoch,
        #             "valid_auc": best_valid_auc,
        #         },
        #         checkpoint_name,
        #     )
        #
        #     print(
        #         f"[Best checkpoint saved] epoch={best_epoch}, "
        #         f"valid_auc={best_valid_auc:.4f}, path={checkpoint_name}"
        #     )

        stop_time = time.time()
        print("time is:{:.4f}s".format(stop_time - start_time))

    # =============================================================================
    # Post-training: best epoch by validation AUC
    # =============================================================================

    valid_auc_array = np.array(valid_auc_roc_list, dtype=float)

    if np.all(np.isnan(valid_auc_array)):
        print("[Warning] All validation AUC values are NaN. Use final epoch as best epoch.")
        max_index_in_valid = epochs - 1
    else:
        max_index_in_valid = int(np.nanargmax(valid_auc_array))

    best_valid_accuracy = valid_accuracy_list[max_index_in_valid]
    best_valid_auc_roc = valid_auc_roc_list[max_index_in_valid]
    best_test_accuracy = test_accuracy_list[max_index_in_valid]
    best_test_auc_roc = test_auc_roc_list[max_index_in_valid]

    # ── CSV export: compact best metrics file ────────────────────────────────
    df = pd.DataFrame(
        {
            "Seed": [args.random_seed],
            "target_task": [getattr(args, "target_task", -1)],
            "Best Validation Accuracy": [best_valid_accuracy],
            "Best Validation AUC ROC": [best_valid_auc_roc],
            "Best Test Accuracy": [best_test_accuracy],
            "Best Test AUC ROC": [best_test_auc_roc],
        }
    ).set_index("Seed")

    csv_file = args.key + "_key_" + args.dataset + "_multilabel_best_metrics.csv"

    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file, index_col="Seed")
        combined_df = pd.concat([existing_df, df])
        combined_df.to_csv(csv_file)
    else:
        df.to_csv(csv_file)

    # =============================================================================
    # Save train / valid / test metrics for every epoch
    # =============================================================================

    epoch_df = pd.DataFrame(
        {
            "epoch": list(range(epochs)),

            "train_loss": train_loss_list,
            "train_acc": train_accuracy_list,
            "train_auc_roc": train_auc_roc_list,

            "valid_loss": valid_loss_list,
            "valid_acc": valid_accuracy_list,
            "valid_auc_roc": valid_auc_roc_list,

            "test_loss": test_loss_list,
            "test_acc": test_accuracy_list,
            "test_auc_roc": test_auc_roc_list,
        }
    )

    csv_name = f"{args.model_name}.job{job_id}_task{task_id}.multilabel_epoch_metrics.csv"
    csv_path = os.path.join("results", csv_name)

    epoch_df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    print(
        f"Checkpoint saved: best epoch {best_epoch}, "
        f"valid AUC={best_valid_auc_roc:.4f}, "
        f"test AUC={best_test_auc_roc:.4f}"
    )

    writer.close()

    # =============================================================================
    # Store best result selected by validation AUC into one master CSV
    # =============================================================================

    best_epoch = max_index_in_valid

    best_row = {
        # identifiers
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "target_task": getattr(args, "target_task", -1),
        "model_name": getattr(args, "model_name", "no_model_name"),
        "key": args.key,
        "seed": args.random_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "freeze_epoch": args.freeze_epoch,

        "enabled_modality": " ".join(args.enabled_modality)
        if isinstance(args.enabled_modality, (list, tuple))
        else str(args.enabled_modality),

        "fuse_mechanism": str(args.fuse_mechanism),
        "strategy": str(getattr(args, "strategy", "")),

        # Slurm ids
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),

        # selection info
        "best_epoch_by": "valid_auc_roc",
        "best_epoch": best_epoch,

        # train metrics at best validation epoch
        "train_loss": float(train_loss_list[best_epoch]),
        "train_acc": float(train_accuracy_list[best_epoch]),
        "train_auc_roc": float(train_auc_roc_list[best_epoch]),

        # validation metrics at best validation epoch
        "valid_loss": float(valid_loss_list[best_epoch]),
        "valid_acc": float(valid_accuracy_list[best_epoch]),
        "valid_auc_roc": float(valid_auc_roc_list[best_epoch]),

        # test metrics at best validation epoch
        "test_loss": float(test_loss_list[best_epoch]),
        "test_acc": float(test_accuracy_list[best_epoch]),
        "test_auc_roc": float(test_auc_roc_list[best_epoch]),
    }

    best_df = pd.DataFrame([best_row])

    os.makedirs("results_bestValidation", exist_ok=True)

    master_csv = os.path.join(
        "results_bestValidation",
        f"{args.dataset}_{args.key}_MASTER_multilabel_best_by_valid_auc.csv",
    )

    if os.path.exists(master_csv):
        best_df.to_csv(master_csv, mode="a", header=False, index=False)
    else:
        best_df.to_csv(master_csv, mode="w", header=True, index=False)

    print(f"[Saved/Append] {master_csv}")
    print(
        f"[Best epoch] {best_epoch} | "
        f"valid_auc={best_row['valid_auc_roc']:.4f} | "
        f"test_auc={best_row['test_auc_roc']:.4f}"
    )