"""Train, evaluate and explain a churn model.

Pipeline: clean -> train/test split (stratified) -> compare a baseline,
logistic regression and a random forest with 5-fold CV -> pick the best by
ROC AUC -> tune the decision threshold -> write metrics, charts and a
one-page summary for a non-technical manager.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls_90d", "senior_citizen"]
CATEGORICAL = ["contract", "internet_service", "tech_support", "payment_method", "paperless_billing"]
TARGET = "churned"
SEED = 42

INK = "#1f2a37"
MUTED = "#8a94a3"
ACCENT = "#2a6fdb"
WARN = "#d9622b"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # New customers with no billing yet: total = one month's charge
    df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"])
    df = df.drop_duplicates("customer_id")
    return df


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def candidates() -> dict[str, Pipeline]:
    return {
        "Baseline (always 'stays')": Pipeline([("pre", preprocessor()), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "Logistic regression": Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(max_iter=1000, C=1.0))]),
        "Random forest": Pipeline(
            [("pre", preprocessor()), ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=-1))]
        ),
    }


def best_threshold(y_true, proba) -> float:
    """Threshold that maximizes F1: balances catching churners vs. false alarms."""
    grid = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y_true, proba >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK)


def plot_drivers(names, coefs, path: Path):
    order = np.argsort(np.abs(coefs))[-10:]
    names, coefs = np.array(names)[order], coefs[order]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [WARN if c > 0 else ACCENT for c in coefs]
    ax.barh(names, coefs, color=colors)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Effect on churn odds (standardized log-odds)", color=INK)
    ax.set_title("Top churn drivers: orange raises risk, blue lowers it", loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(results, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in results.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="Random guess")
    ax.set_xlabel("False alarm rate", color=INK)
    ax.set_ylabel("Share of churners caught", color=INK)
    ax.set_title("ROC curve on held-out customers", loc="left", color=INK)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    labels = [["Correctly kept", "False alarm"], ["Missed churner", "Caught churner"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["Predicted stays", "Predicted churns"])
    ax.set_yticks([0, 1], ["Actually stayed", "Actually churned"])
    ax.set_title("Where the model is right and wrong", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plain_name(feature: str) -> str:
    f = feature.replace("num__", "").replace("cat__", "")
    mapping = {
        "tenure_months": "Longer tenure",
        "monthly_charges": "Higher monthly bill",
        "total_charges": "Higher lifetime spend",
        "support_calls_90d": "More support calls (90 days)",
        "senior_citizen": "Senior citizen",
    }
    if f in mapping:
        return mapping[f]
    col, _, val = f.partition("_")
    for c in CATEGORICAL:
        if f.startswith(c + "_"):
            col, val = c, f[len(c) + 1:]
    return f"{col.replace('_', ' ').capitalize()}: {val}"


def run(data_path: str, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = clean(pd.read_csv(data_path))
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    comparison, roc_data, fitted = {}, {}, {}
    for name, pipe in candidates().items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_data[name] = (fpr, tpr, auc)
        comparison[name] = {
            "cv_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_auc_std": round(float(cv_auc.std()), 4),
            "test_auc": round(float(auc), 4),
            "test_accuracy": round(float(accuracy_score(y_te, proba >= 0.5)), 4),
        }
        fitted[name] = pipe

    # Choose between real models by CV AUC; prefer logistic regression if within 0.01 (explainable)
    lr, rf = comparison["Logistic regression"], comparison["Random forest"]
    chosen = "Logistic regression" if lr["cv_auc_mean"] >= rf["cv_auc_mean"] - 0.01 else "Random forest"
    model = fitted[chosen]

    # Tune threshold on training data via CV-free split to avoid touching test set
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=SEED)
    tuner = candidates()[chosen].fit(X_fit, y_fit)
    threshold = best_threshold(y_val, tuner.predict_proba(X_val)[:, 1])

    proba = model.predict_proba(X_te)[:, 1]
    pred = proba >= threshold
    cm = confusion_matrix(y_te, pred)
    metrics = {
        "chosen_model": chosen,
        "decision_threshold": round(threshold, 2),
        "test_rows": int(len(y_te)),
        "churn_rate": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "model_comparison": comparison,
    }

    names = [plain_name(n) for n in model.named_steps["pre"].get_feature_names_out()]
    if chosen == "Logistic regression":
        coefs = model.named_steps["clf"].coef_[0]
    else:
        coefs = model.named_steps["clf"].feature_importances_
    plot_drivers(names, coefs, out / "churn_drivers.png")
    plot_roc(roc_data, out / "roc_curve.png")
    plot_confusion(cm, out / "confusion_matrix.png")

    ranked = sorted(zip(names, coefs), key=lambda t: -t[1])
    metrics["top_risk_drivers"] = [n for n, c in ranked[:3]]
    metrics["top_protective_factors"] = [n for n, c in ranked[::-1][:3]]

    # Score the full customer base and export a call list for the retention team
    df["churn_risk"] = model.predict_proba(X)[:, 1].round(3)
    at_risk = df[df.churned == 0].sort_values("churn_risk", ascending=False).head(200)
    at_risk[["customer_id", "churn_risk", "contract", "tenure_months", "monthly_charges"]].to_csv(
        out / "retention_call_list.csv", index=False
    )

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_summary(metrics, cm, out / "SUMMARY.md")
    return metrics


def write_summary(m: dict, cm, path: Path):
    tn, fp, fn, tp = cm.ravel()
    caught = tp / (tp + fn)
    lines = [
        "# Churn model: one-page summary",
        "",
        "*Synthetic data. Numbers below come from the latest run of `python run.py`.*",
        "",
        "## The answer",
        "",
        f"The model catches **{caught:.0%} of customers who are about to cancel**, "
        f"with **{m['precision']:.0%}** of its flags being real churners "
        f"(overall accuracy {m['accuracy']:.0%}, ROC AUC {m['roc_auc']:.2f}).",
        "",
        "## What drives cancellations",
        "",
        "Raises risk: " + ", ".join(m["top_risk_drivers"]) + ".",
        "",
        "Lowers risk: " + ", ".join(m["top_protective_factors"]) + ".",
        "",
        "![Drivers](churn_drivers.png)",
        "",
        "## What I'd do with this",
        "",
        "1. Call the 200 highest-risk current customers (`retention_call_list.csv`) and offer a contract upgrade.",
        "2. Target month-to-month customers in their first year, where risk is highest.",
        "3. Re-run monthly and track whether flagged customers who got a call churned less.",
        "",
        "## How good is it, honestly?",
        "",
        f"Out of {m['test_rows']:,} held-out customers: {tp} churners caught, {fn} missed, "
        f"{fp} false alarms. A baseline that predicts 'everyone stays' scores "
        f"{m['model_comparison']['Baseline (always ' + chr(39) + 'stays' + chr(39) + ')']['test_accuracy']:.0%} accuracy "
        "but catches zero churners, which is why accuracy alone is misleading here.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/customers.csv")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    m = run(a.data, a.out)
    print(json.dumps({k: v for k, v in m.items() if k != "model_comparison"}, indent=2))


if __name__ == "__main__":
    main()
