import os
import time

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
    print("==================================")

    model = Net(
        n_output_layers=number_of_task,
        fusion=fusion,
    ).to(device)

    return model


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def model_eval(args, model, device, loader, num_sample, tokenizer, criterion):
    """
    Evaluate model on a data loader.
    """
    number_of_task = args.n_tasks
    eval_loss = 0.0
    preds = []
    labels = []

    model.eval()

    for data in tqdm(loader):
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

        y = torch.as_tensor(
            data["target"],
            device=device,
            dtype=torch.float32,
        )

        if outputs.ndim == 2 and outputs.size(1) == 1 and y.ndim == 1:
            y = y.unsqueeze(1)

        eval_loss += criterion(outputs.float(), y).item()

        preds += torch.sigmoid(outputs).view(-1).detach().cpu().tolist()
        labels += y.detach().cpu().tolist()

    trues = np.array(labels).reshape(-1, number_of_task).T
    belief_scores = np.array(preds).reshape(-1, number_of_task).T

    roc_auc_score_list = [
        roc_auc_score(trues[i].tolist(), belief_scores[i].tolist())
        for i in range(number_of_task)
    ]

    auc_roc = float(sum(roc_auc_score_list) / number_of_task)
    accuracy = accuracy_score(labels, [1 if v > 0.5 else 0 for v in preds])

    eval_loss = eval_loss / num_sample

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


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = sd.parse_input()

    sd.set_global_seed(args.random_seed)

    epochs = args.epochs
    number_of_task = args.n_tasks
    fusion_end_epochs = args.freeze_epoch

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
    criterion = torch.nn.BCEWithLogitsLoss()

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

    # ── Track best model on disk, not in CPU memory ──────────────────────────
    best_valid_auc = -1.0
    best_epoch = 0

    checkpoint_dir = "./checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = f"{checkpoint_dir}/3d_branch_{args.dataset}_{args.model_name}.pth"

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
        preds = []
        labels = []

        for data in tqdm(train_loader):
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

            y = torch.as_tensor(
                data["target"],
                device=device,
                dtype=torch.float32,
            )

            if outputs.ndim == 2 and outputs.size(1) == 1 and y.ndim == 1:
                y = y.unsqueeze(1)

            loss = criterion(outputs.float(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            preds += torch.sigmoid(outputs).view(-1).detach().cpu().tolist()
            labels += y.detach().cpu().tolist()

        scheduler.step()

        # ── Train metrics ────────────────────────────────────────────────────
        train_loss = train_loss_sum / train_size
        print("Train/Loss", train_loss, epoch)

        trues = np.array(labels).reshape(-1, number_of_task).T
        belief_scores = np.array(preds).reshape(-1, number_of_task).T

        roc_auc_score_list = [
            roc_auc_score(trues[i].tolist(), belief_scores[i].tolist())
            for i in range(number_of_task)
        ]

        train_auc_roc = float(sum(roc_auc_score_list) / number_of_task)
        train_accuracy = accuracy_score(
            labels,
            [1 if v > 0.5 else 0 for v in preds],
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

        # # ── Snapshot best model weights in memory ────────────────────────────
        # if valid_auc_roc > best_valid_auc:
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

    max_index_in_valid = valid_auc_roc_list.index(max(valid_auc_roc_list))

    best_valid_accuracy = valid_accuracy_list[max_index_in_valid]
    best_valid_auc_roc = valid_auc_roc_list[max_index_in_valid]
    best_test_accuracy = test_accuracy_list[max_index_in_valid]
    best_test_auc_roc = test_auc_roc_list[max_index_in_valid]

    # ── CSV export: compact best metrics file ────────────────────────────────
    df = pd.DataFrame(
        {
            "Seed": [args.random_seed],
            "Best Validation Accuracy": [best_valid_accuracy],
            "Best Validation AUC ROC": [best_valid_auc_roc],
            "Best Test Accuracy": [best_test_accuracy],
            "Best Test AUC ROC": [best_test_auc_roc],
        }
    ).set_index("Seed")

    csv_file = args.key + "_key_" + args.dataset + "_best_metrics.csv"

    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file, index_col="Seed")
        combined_df = pd.concat([existing_df, df])
        combined_df.to_csv(csv_file)
    else:
        df.to_csv(csv_file)

    # =============================================================================
    # Added function 1:
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

    csv_name = f"{args.model_name}.job{job_id}_task{task_id}.epoch_metrics.csv"
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
    # Added function 2:
    # Store best result selected by validation AUC into one master CSV
    # =============================================================================

    best_epoch = int(np.argmax(valid_auc_roc_list))

    best_row = {
        # identifiers
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "model_name": getattr(args, "model_name", "no_model_name"),
        "key": args.key,
        "seed": args.random_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,

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
        f"{args.dataset}_{args.key}_MASTER_best_by_valid_auc.csv",
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