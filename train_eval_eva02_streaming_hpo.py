import time
import re
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# ---------------- Config ----------------
BASE = Path(__file__).resolve().parent

TRAIN_CSV = BASE / "train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"
VAL_CSV   = BASE / "val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"
V2_CSV    = BASE / "v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"

SCALER_P  = BASE / "eva02_scaler.joblib"
MODEL_P   = BASE / "eva02_sgd_logreg.joblib"

SUMMARY_P = BASE / "eva02_results_summary.csv"
PERCLASS_P= BASE / "eva02_per_class_gap.csv"
SHIFT_P   = BASE / "eva02_feature_shift_val_vs_v2.csv"
VAL_PREDS = BASE / "eva02_val_preds.csv"
V2_PREDS  = BASE / "eva02_v2_preds.csv"

CHUNK         = 200_000      # reduce if RAM is tight
HOLDOUT_FRAC  = 0.10         # 10% of train reserved for internal validation (tuning)
NUM_CLASSES   = 1000
HPO_ALPHAS    = [1e-4, 5e-4, 1e-3]
HPO_EPOCHS    = [1, 2, 3]
RANDOM_STATE  = 42
# ----------------------------------------


# ---------- helpers: columns, dtypes, streaming ----------
def infer_feature_cols(csv_path: Path):
    head = pd.read_csv(csv_path, nrows=0)
    cols = [str(c).strip() for c in head.columns]
    feat_idx = sorted([int(c) for c in cols if re.fullmatch(r"\d+", c)])
    feat_cols = [str(i) for i in feat_idx]
    assert "label" in cols, "Expected a 'label' column."
    return ["label"] + feat_cols, feat_cols

def dtypes_for(usecols, feat_cols):
    dt = {"label": "int16"}
    for c in feat_cols: dt[c] = "float32"
    return dt

def stream_chunks(path: Path, usecols, dtypes, chunksize=CHUNK):
    return pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize)

# deterministic internal split: 10% holdout by hashed global row index
def in_holdout(global_idx: int, frac: float = HOLDOUT_FRAC) -> bool:
    h = int(hashlib.md5(str(global_idx).encode()).hexdigest(), 16)
    return (h % 10_000) < int(frac * 10_000)


# ---------- scaler passes ----------
def fit_scaler_on_train_only(train_csv, feat_cols, usecols, dtypes):
    scaler = StandardScaler(with_mean=True, with_std=True)
    t0 = time.time()
    n_rows, gidx = 0, 0
    for chunk in stream_chunks(train_csv, usecols, dtypes):
        # keep only non-holdout rows
        mask = np.fromiter((not in_holdout(gidx + i) for i in range(len(chunk))), dtype=bool)
        gidx += len(chunk)
        if not mask.any(): 
            continue
        X = chunk.loc[mask, feat_cols].values
        scaler.partial_fit(X)
        n_rows += X.shape[0]
    elapsed = time.time() - t0
    print(f"[Scaler] partial_fit on ~{n_rows:,} train-only rows in {elapsed:.1f}s")
    return scaler, elapsed

def fit_scaler_on_full_train(train_csv, feat_cols, usecols, dtypes):
    scaler = StandardScaler(with_mean=True, with_std=True)
    t0 = time.time()
    n_rows = 0
    for chunk in stream_chunks(train_csv, usecols, dtypes):
        X = chunk[feat_cols].values
        scaler.partial_fit(X)
        n_rows += len(chunk)
    elapsed = time.time() - t0
    print(f"[Scaler-final] partial_fit on full train (~{n_rows:,}) in {elapsed:.1f}s")
    return scaler, elapsed


# ---------- streamed evaluation/prediction ----------
def streamed_accuracy(csv_path, scaler, clf, feat_cols, usecols, dtypes):
    correct, total = 0, 0
    t0 = time.time()
    for ch in stream_chunks(csv_path, usecols, dtypes):
        X = scaler.transform(ch[feat_cols].values)
        y = ch["label"].values
        pred = clf.predict(X)
        correct += (pred == y).sum()
        total += y.size
    return (correct / total), (time.time() - t0)

def streamed_preds(csv_path, scaler, clf, feat_cols, usecols, dtypes):
    ys, yhats = [], []
    for ch in stream_chunks(csv_path, usecols, dtypes):
        X = scaler.transform(ch[feat_cols].values)
        ys.append(ch["label"].values)
        yhats.append(clf.predict(X))
    return np.concatenate(ys), np.concatenate(yhats)


# ---------- HPO: train/eval on internal split ----------
def train_on_train_only(train_csv, scaler, alpha, epochs, feat_cols, usecols, dtypes):
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
    )
    classes = np.arange(NUM_CLASSES, dtype=np.int32)
    first = True
    t0 = time.time()
    for ep in range(epochs):
        gidx = 0
        n_rows = 0
        for ch in stream_chunks(train_csv, usecols, dtypes):
            mask = np.fromiter((not in_holdout(gidx + i) for i in range(len(ch))), dtype=bool)
            gidx += len(ch)
            if not mask.any(): 
                continue
            X = scaler.transform(ch.loc[mask, feat_cols].values)
            y = ch.loc[mask, "label"].values
            if first:
                clf.partial_fit(X, y, classes=classes); first = False
            else:
                clf.partial_fit(X, y)
            n_rows += X.shape[0]
        print(f"[HPO] alpha={alpha}, epochs={ep+1}/{epochs} trained ~{n_rows:,} rows")
    train_time = time.time() - t0
    return clf, train_time

def eval_on_holdout(train_csv, scaler, clf, feat_cols, usecols, dtypes):
    correct, total = 0, 0
    t0 = time.time()
    gidx = 0
    for ch in stream_chunks(train_csv, usecols, dtypes):
        mask = np.fromiter((in_holdout(gidx + i) for i in range(len(ch))), dtype=bool)
        gidx += len(ch)
        if not mask.any(): 
            continue
        X = scaler.transform(ch.loc[mask, feat_cols].values)
        y = ch.loc[mask, "label"].values
        pred = clf.predict(X)
        correct += (pred == y).sum()
        total += y.size
    return (correct / total), (time.time() - t0)


# ---------- final training on full training set ----------
def train_on_full_train(train_csv, scaler, alpha, epochs, feat_cols, usecols, dtypes):
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
    )
    classes = np.arange(NUM_CLASSES, dtype=np.int32)
    first = True
    t0 = time.time()
    for ep in range(epochs):
        n_rows = 0
        for ch in stream_chunks(train_csv, usecols, dtypes):
            X = scaler.transform(ch[feat_cols].values)
            y = ch["label"].values
            if first:
                clf.partial_fit(X, y, classes=classes); first = False
            else:
                clf.partial_fit(X, y)
            n_rows += len(y)
        print(f"[FINAL] alpha={alpha}, epochs={ep+1}/{epochs} trained ~{n_rows:,} rows")
    train_time = time.time() - t0
    return clf, train_time


# ------------------------ main ------------------------
def main():
    print("=== EVA-02: streaming HPO + final train ===")
    usecols, feat_cols = infer_feature_cols(TRAIN_CSV)
    dtypes = dtypes_for(usecols, feat_cols)
    feat_dim = len(feat_cols)
    print(f"Inferred feature dim: {feat_dim}")

    # ----- HPO: scaler on train-only, then small grid over (alpha, epochs)
    scaler_tune, t_scale_tune = fit_scaler_on_train_only(TRAIN_CSV, feat_cols, usecols, dtypes)

    best = {"alpha": None, "epochs": None, "val_acc": -1.0, "train_time": 0.0}
    for alpha in HPO_ALPHAS:
        for epochs in HPO_EPOCHS:
            clf_tmp, t_train = train_on_train_only(TRAIN_CSV, scaler_tune, alpha, epochs, feat_cols, usecols, dtypes)
            acc_holdout, t_hold = eval_on_holdout(TRAIN_CSV, scaler_tune, clf_tmp, feat_cols, usecols, dtypes)
            print(f"[HPO] alpha={alpha} epochs={epochs} → internal_val_acc={acc_holdout*100:.2f}%")
            if acc_holdout > best["val_acc"]:
                best.update({"alpha": alpha, "epochs": epochs, "val_acc": acc_holdout, "train_time": t_train})

    print(f"[HPO] Best: alpha={best['alpha']}, epochs={best['epochs']}, "
          f"internal_val_acc={best['val_acc']*100:.2f}%")

    # ----- Final training: scaler on FULL train, then train full model with best hyperparams
    scaler_final, t_scale_final = fit_scaler_on_full_train(TRAIN_CSV, feat_cols, usecols, dtypes)
    clf_final, t_train_final = train_on_full_train(TRAIN_CSV, scaler_final, best["alpha"], best["epochs"],
                                                   feat_cols, usecols, dtypes)

    dump(scaler_final, SCALER_P); dump(clf_final, MODEL_P)
    print(f"Saved artifacts: {SCALER_P.name}, {MODEL_P.name}")

    # ----- Evaluate on official Test set 1 & Test set 2
    acc_val, t_val = streamed_accuracy(VAL_CSV, scaler_final, clf_final, feat_cols, usecols, dtypes)
    acc_v2,  t_v2  = streamed_accuracy(V2_CSV,  scaler_final, clf_final, feat_cols, usecols, dtypes)
    print(f"[EVAL] Test set 1 (val): {acc_val*100:.2f}%  (in {t_val:.1f}s)")
    print(f"[EVAL] Test set 2 (v2):  {acc_v2*100:.2f}%  (in {t_v2:.1f}s)")

    # ----- Save predictions (for analysis)
    y1, p1 = streamed_preds(VAL_CSV, scaler_final, clf_final, feat_cols, usecols, dtypes)
    y2, p2 = streamed_preds(V2_CSV,  scaler_final, clf_final, feat_cols, usecols, dtypes)
    pd.DataFrame({"y": y1, "yhat": p1}).to_csv(VAL_PREDS, index=False)
    pd.DataFrame({"y": y2, "yhat": p2}).to_csv(V2_PREDS, index=False)
    print(f"Saved preds: {VAL_PREDS.name}, {V2_PREDS.name}")

    # ----- Per-class gap
    rows = []
    for c in range(NUM_CLASSES):
        m1 = (y1 == c); m2 = (y2 == c)
        if m1.any() and m2.any():
            acc1 = (p1[m1] == c).mean()
            acc2 = (p2[m2] == c).mean()
            rows.append({"class": c, "acc_val": acc1, "acc_v2": acc2, "drop": acc1 - acc2})
    pd.DataFrame(rows).sort_values("drop", ascending=False).to_csv(PERCLASS_P, index=False)
    print(f"Saved: {PERCLASS_P.name}")

    # ----- Feature shift (val vs v2): standardized mean difference per feature
    def streamed_mean_var(csv_path):
        n = 0
        mean = None
        M2 = None
        for ch in stream_chunks(csv_path, usecols, dtypes):
            X = ch[feat_cols].values
            if mean is None:
                mean = X.mean(axis=0)
                diff = X - mean
                M2 = (diff * diff).sum(axis=0)
                n = X.shape[0]
            else:
                n_old = n
                n += X.shape[0]
                delta = X.mean(axis=0) - mean
                mean = mean + delta * (X.shape[0] / n)
                M2 = M2 + ((X - mean) ** 2).sum(axis=0) + (delta ** 2) * (n_old * X.shape[0] / n)
        var = M2 / max(n - 1, 1)
        return mean, np.sqrt(var)

    m1, s1 = streamed_mean_var(VAL_CSV)
    m2, s2 = streamed_mean_var(V2_CSV)
    shift = np.abs(m1 - m2) / (0.5 * (s1 + s2) + 1e-8)
    pd.DataFrame({"feat": feat_cols, "std_shift": shift}).sort_values("std_shift", ascending=False)\
        .to_csv(SHIFT_P, index=False)
    print(f"Saved: {SHIFT_P.name}")

    # ----- summary row for the report
    pd.DataFrame([{
        "model": "EVA-02 SGD logreg (streaming, HPO on train-holdout)",
        "feat_dim": len(feat_cols),
        "best_alpha": best["alpha"],
        "best_epochs": best["epochs"],
        "internal_val_acc": round(best["val_acc"], 4),
        "final_val_acc": round(acc_val, 4),
        "final_v2_acc": round(acc_v2, 4),
        "time_scaler_tune_s": round(t_scale_tune, 1),
        "time_scaler_final_s": round(t_scale_final, 1),
        "time_train_final_s": round(t_train_final, 1),
        "time_eval_s": round(t_val + t_v2, 1),
    }]).to_csv(SUMMARY_P, index=False)
    print(f"Saved: {SUMMARY_P.name}")

if __name__ == "__main__":
    main()
