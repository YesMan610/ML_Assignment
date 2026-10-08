"""
predict_poly_regression.py
============================
Polynomial Regression assignment — PREDICTION script.

Takes the `model_config.json` produced by train_poly_regression.py (which
contains the learned weight vector W, bias, degree, feature list and
standardisation stats) and applies it to a test CSV, entirely in PyTorch.

USAGE
-----
    python predict_poly_regression.py \
        --test_csv BT2024001_test_var1.csv \
        --config results_var1/model_config.json \
        --out_csv BT2024001_pred_var1.csv

If your test CSV also happens to contain the true target column (e.g. you
carved out your own held-out test split to sanity-check before submitting),
pass --target y and the script will also print MSE / R^2 and save a
predicted-vs-actual diagnostic plot. The real hidden test set from the
assignment will NOT have this column, which is fine -- --target is optional.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--config", required=True, help="model_config.json from training")
    ap.add_argument("--out_csv", required=True, help="where to write predictions")
    ap.add_argument("--target", default=None,
                     help="only set this if your test CSV has true y values you want to check against")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    features = cfg["features"]
    degree = cfg["degree"]
    scaler_mean = np.array(cfg["scaler_mean"], dtype=np.float64)
    scaler_scale = np.array(cfg["scaler_scale"], dtype=np.float64)
    W = torch.tensor(cfg["weight"], dtype=torch.float32).view(-1, 1)   # (n_terms, 1)
    b = torch.tensor(cfg["bias"], dtype=torch.float32)

    df = pd.read_csv(args.test_csv)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"test CSV is missing feature columns required by the model: {missing}")
    X_raw = df[features].values.astype(np.float64)

    # Rebuild the SAME polynomial feature layout used in training.
    # PolynomialFeatures has no learned parameters -- it's a fixed structural
    # transform determined only by `degree` and the number/order of input
    # features, so re-fitting it here on the test data's shape is safe and
    # reproduces the exact same columns training used.
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_raw)

    if X_poly.shape[1] != len(scaler_mean):
        raise ValueError(
            f"Feature mismatch: test produces {X_poly.shape[1]} polynomial terms, "
            f"but the saved model expects {len(scaler_mean)}. Check that --features "
            f"and the column order match what training used."
        )

    X_scaled = (X_poly - scaler_mean) / scaler_scale

    # ---- prediction, in PyTorch ----
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred = (X_t @ W + b).numpy().ravel()

    target_col = cfg.get("target", "y")
    out_df = df.copy()
    out_df[target_col] = y_pred
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} predictions to {args.out_csv}")
    print("NOTE: double-check the sample submission file for the exact expected "
          "column name/order before you submit -- this script writes all original "
          f"test columns plus a '{target_col}' column of predictions.")

    # ---- optional: evaluate against ground truth, if you provided one ----
    if args.target and args.target in df.columns:
        y_true = df[args.target].values.astype(np.float64)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\nEvaluation against provided '{args.target}' column:")
        print(f"  MSE = {mse:.4f}")
        print(f"  R2  = {r2:.4f}")

        out_dir = os.path.dirname(os.path.abspath(args.out_csv)) or "."
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.scatter(y_true, y_pred, alpha=0.4, s=15)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax1.plot(lims, lims, "k--", lw=1)
        ax1.set_xlabel("actual y")
        ax1.set_ylabel("predicted y")
        ax1.set_title(f"Test set: predicted vs actual (R2={r2:.3f})")

        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.4, s=15)
        ax2.axhline(0, color="k", ls="--", lw=1)
        ax2.set_xlabel("predicted y")
        ax2.set_ylabel("residual")
        ax2.set_title(f"Test set residuals (MSE={mse:.3f})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "test_set_diagnostics.png"), dpi=140)
        plt.close(fig)
        print(f"Saved test_set_diagnostics.png to {out_dir}")


if __name__ == "__main__":
    main()
