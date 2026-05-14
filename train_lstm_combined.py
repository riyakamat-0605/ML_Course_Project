

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ─────────────────────────────────────────────
# CONFIG  — change only this section if needed
# ─────────────────────────────────────────────
SEQUENCE_LENGTH = 30
FEATURE_DIM     = 127          # 1 (two-hand flag) + 63 (left) + 63 (right)
TEST_SIZE       = 0.2
RANDOM_SEED     = 42
EPOCHS          = 50           # EarlyStopping will cut this short if needed
BATCH_SIZE      = 16

USER_DATA_DIR   = "data/sequences"       # subfolders named by label string
CSV_DATA_DIR    = "data/sequences_csv"   # subfolders named by class integer id

MODEL_SAVE_PATH = "models/isl_lstm_combined.h5"
BEST_CKPT_PATH  = "models/best_checkpoint.h5"  # best val_accuracy during training

# ─────────────────────────────────────────────
# LABEL MAP  — single source of truth
#   key   = folder name under data/sequences/
#   value = integer class id
# ─────────────────────────────────────────────
LABEL_MAP = {
    "hello":  0,
    "thanks": 1,
    "yes":    2,
}

# Inverse map used for reports
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES   = len(LABEL_MAP)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_sequences_from_folder(folder: str, class_id: int,
                                sequences: list, labels: list,
                                source_tag: str = "") -> int:
    """
    Load all .npy files from `folder`, validate shape,
    and append to sequences / labels lists.
    Returns count of sequences loaded.
    """
    if not os.path.isdir(folder):
        print(f"  [SKIP] Folder not found: {folder}")
        return 0

    count = 0
    for fname in os.listdir(folder):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(folder, fname)
        try:
            seq = np.load(path)
        except Exception as e:
            print(f"  [WARN] Could not load {path}: {e}")
            continue

        if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            print(f"  [WARN] Bad shape {seq.shape} in {path} — skipped")
            continue

        sequences.append(seq)
        labels.append(class_id)
        count += 1

    return count


def build_model(num_classes: int) -> Sequential:
    """
    Two-layer LSTM classifier.
    Input shape: (SEQUENCE_LENGTH, FEATURE_DIM)
    """
    model = Sequential(
        [
            LSTM(128, return_sequences=True,
                 input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ],
        name="ISL_LSTM",
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ISL LSTM — Training Curves", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix (normalised)", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def save_classification_report(y_true, y_pred,
                                class_names: list, save_path: str,
                                test_acc: float, test_loss: float):
    report = classification_report(y_true, y_pred, target_names=class_names)
    lines = [
        "=" * 55,
        "  ISL LSTM — Classification Report",
        "=" * 55,
        f"  Test Accuracy : {test_acc * 100:.2f}%",
        f"  Test Loss     : {test_loss:.4f}",
        "=" * 55,
        "",
        report,
    ]
    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    # Also print to console
    print("\n" + "\n".join(lines))
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs("models", exist_ok=True)

    sequences: list = []
    labels:    list = []

    # ── 1. Load webcam-collected sequences ────────────────────────────────
    print("\n[1/5] Loading user-collected sequences ...")
    user_total = 0
    for label_name, class_id in LABEL_MAP.items():
        folder = os.path.join(USER_DATA_DIR, label_name)
        n = load_sequences_from_folder(folder, class_id, sequences, labels,
                                       source_tag="user")
        print(f"  {label_name:10s} (id={class_id})  →  {n} sequences")
        user_total += n
    print(f"  User data total: {user_total} sequences")

    # ── 2. Load CSV-converted sequences ──────────────────────────────────
    print("\n[2/5] Loading CSV-converted sequences ...")
    csv_total = 0
    for label_name, class_id in LABEL_MAP.items():
        folder = os.path.join(CSV_DATA_DIR, str(class_id))
        n = load_sequences_from_folder(folder, class_id, sequences, labels,
                                       source_tag="csv")
        print(f"  class {class_id} ({label_name:8s})  →  {n} sequences")
        csv_total += n
    print(f"  CSV data total: {csv_total} sequences")

    # ── 3. Sanity check ──────────────────────────────────────────────────
    if len(sequences) == 0:
        raise RuntimeError(
            "No sequences loaded. "
            "Run collect_sequence.py or convert_csv_to_sequences.py first."
        )

    X = np.array(sequences, dtype=np.float32)   # (N, 30, 127)
    y_int = np.array(labels, dtype=np.int32)     # (N,)
    y_cat = to_categorical(y_int, num_classes=NUM_CLASSES)

    print(f"\n  Dataset shape  : {X.shape}")
    print(f"  Classes present: {np.unique(y_int)}")
    for cid in np.unique(y_int):
        print(f"    class {cid} ({INV_LABEL_MAP[cid]}): {np.sum(y_int == cid)} samples")

    # ── 4. Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_int,          # keep class balance in both splits
    )
    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── 5. Build model ────────────────────────────────────────────────────
    print("\n[3/5] Building model ...")
    model = build_model(NUM_CLASSES)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=BEST_CKPT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── 6. Train ──────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for up to {EPOCHS} epochs "
          f"(EarlyStopping patience=10) ...\n")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # ── 7. Evaluate & save artefacts ──────────────────────────────────────
    print("\n[5/5] Evaluating and saving artefacts ...")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Predictions for reports
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test,       axis=1)
    class_names = [INV_LABEL_MAP[i] for i in range(NUM_CLASSES)]

    plot_training_curves(history,
                         save_path="models/training_curves.png")

    plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path="models/confusion_matrix.png")

    save_classification_report(y_true, y_pred, class_names,
                                save_path="models/classification_report.txt",
                                test_acc=test_acc, test_loss=test_loss)

    # Save final model in .h5 format (matches realtime_lstm.py loader)
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Final model saved → {MODEL_SAVE_PATH}")
    print(f"✅ Best checkpoint   → {BEST_CKPT_PATH}")
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()