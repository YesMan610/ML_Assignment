"""
train_poly_regression.py
=========================
Polynomial Regression assignment — TRAINING script (finds the weight vector W).

What this script does, in order (matches the flow you asked for):

  1. Loads your training CSV and uses shuffled K-fold cross-validation for model selection
     (you don't have the hidden test labels, so this is how we estimate
     generalisation before submitting).
  2. FEATURE DEPENDENCY PLOTS — for every raw input feature, plots that
     feature against the target y, so you can see each parameter's raw
     effect on the data before any model is fit.
  3. TERM-IMPORTANCE CHECK (before picking L1 vs L2) — expands to the
     richest polynomial (max_degree), and plots each polynomial term's raw
     correlation with the target. This is the "how spread out is the effect"
     diagnostic: if only a handful of terms carry real signal, that's a
     sparsity hint favouring L1; if the effect is spread thinly across many
     terms, that favours L2. It's only a hint printed to the console though
     — step 4 below decides for real, using actual cross-validation error.
  4. JOINT DEGREE + REGULARISATION SEARCH — degree and regularisation are
     searched TOGETHER, not degree-first-then-regularisation-second. For
     every candidate degree, the script tries no regularisation, its own
     best L1 lambda, and its own best L2 lambda, so a higher-capacity
     polynomial that would overfit unregularised is still allowed to win if
     L1/L2 tames it and gets a lower validation MSE than any smaller degree
     does. The global best (degree, reg_type, lambda) across the whole grid
     is what gets used. Plots: degree vs. loss without regularisation (for
     context), degree vs. best-achievable loss WITH regularisation (the
     plot that actually matters), L1 vs L2 validation loss at the winning
     degree, and the L1 / L2 coefficient paths at that degree.
  5. Retrains W on the FULL training set with the winning config (more
     epochs, for a clean final fit), and reports final MSE / R^2.
  6. MODEL-EFFECT PLOTS — partial-dependence style plots of the *fitted*
     model: vary one feature at a time (holding the others at their mean)
     and plot the model's predicted y — this is "each feature's effect
     *according to the trained model*", complementing the raw plots in (2).
  7. Saves a single JSON config (+ nothing else) that the prediction script
     needs: feature list, degree, standardisation stats, regularisation
     choice, and W itself (weight vector + bias).

USAGE
-----
    python train_poly_regression.py \
        --train_csv BT2024001_train_var1.csv \
        --target y \
        --features x1 x2 x3 \
        --max_degree 6 \
        --out_dir results_var1

Leave --features out to use every column except --target.
Leave --degree out to let the script pick the best degree automatically
(pass --degree N to force a specific degree and skip the sweep's choice).

Everything is plain PyTorch (torch.optim + autograd) for the actual
optimisation of W. sklearn is used ONLY for two bits of bookkeeping that
have nothing to do with learning W: building the polynomial feature
matrix (PolynomialFeatures) and standardising columns (StandardScaler) —
both are deterministic, non-learned transforms, so re-doing them by hand
in PyTorch would not change any result.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

torch.manual_seed(0)
np.random.seed(0)


# --------------------------------------------------------------------------
# 0. The model itself: y_hat = X @ w + b  (a single linear layer)
# --------------------------------------------------------------------------
class LinearModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


def train_linear_pytorch(X_train, y_train, X_val, y_val,
                          reg_type="none", lam=0.0,
                          epochs=3000, lr=0.02, patience=300):
    """
    Trains y = X @ w + b with plain gradient descent (Adam) in PyTorch.
    reg_type: "none" | "l1" | "l2"  (penalty is applied to the WEIGHTS only,
    never to the bias — penalising the bias would just discourage the model
    from getting the average right, which is never what you want).

    Returns: trained model, train_loss_history, val_loss_history
    """
    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        Xval_t = torch.tensor(X_val, dtype=torch.float32)
        yval_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    model = LinearModel(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    train_hist, val_hist = [], []
    best_val, best_state, bad_epochs = np.inf, None, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(Xtr_t)
        data_loss = mse(pred, ytr_t)

        weight = model.linear.weight
        if reg_type == "l1":
            penalty = lam * weight.abs().sum()
        elif reg_type == "l2":
            penalty = lam * (weight ** 2).sum()
        else:
            penalty = torch.tensor(0.0)

        loss = data_loss + penalty
        loss.backward()
        optimizer.step()

        train_hist.append(data_loss.item())  # log the *unpenalised* MSE so
                                              # curves are comparable across
                                              # reg types / lambdas

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = mse(model(Xval_t), yval_t).item()
            val_hist.append(val_loss)

            # early stopping on validation loss, so huge --epochs is safe
            if val_loss < best_val - 1e-9:
                best_val, best_state, bad_epochs = val_loss, \
                    {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if has_val and best_state is not None:
        model.load_state_dict(best_state)

    return model, train_hist, val_hist


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def build_poly(X_raw, degree, scaler=None, fit_scaler=False):
    """X_raw -> polynomial features (total-degree <= `degree`, no bias
    column since the model's own bias handles that) -> standardised."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_raw)
    if fit_scaler:
        scaler = StandardScaler().fit(X_poly)
    X_scaled = scaler.transform(X_poly)
    return X_scaled, poly, scaler


def evaluate(model, X, y):
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pred = model(Xt).numpy().ravel()
    return mean_squared_error(y, pred), r2_score(y, pred), pred


def term_names(feature_names, degree):
    """Human-readable names for each polynomial term, e.g. x1^2, x1*x3."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(np.zeros((1, len(feature_names))))
    return poly.get_feature_names_out(feature_names)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--target", default="y")
    ap.add_argument("--features", nargs="+", default=None,
                     help="feature columns to use; default = all columns except --target")
    ap.add_argument("--max_degree", type=int, default=8,
                     help="largest degree tried in the degree sweep")
    ap.add_argument("--degree", type=int, default=None,
                     help="force this degree instead of letting the sweep choose one")
    ap.add_argument("--n_folds", type=int, default=5,
                     help="number of shuffled CV folds used for model selection")
    ap.add_argument("--n_lambdas", type=int, default=14,
                     help="how many lambda values to try per (degree, reg_type) in the joint search")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--out_dir", default="poly_reg_results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- 1. load & set up cross-validation ----------------
    df = pd.read_csv(args.train_csv)
    features = args.features or [c for c in df.columns if c != args.target]
    X_all = df[features].values.astype(np.float64)
    y_all = df[args.target].values.astype(np.float64)

    if args.n_folds < 2:
        raise ValueError("--n_folds must be at least 2")
    if args.n_folds > len(X_all):
        raise ValueError("--n_folds cannot exceed the number of training rows")

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    print(f"Loaded {len(df)} rows | features used: {features}")
    print(f"Model selection -> {args.n_folds}-fold cross-validation (shuffle=True, random_state=42)")

    # ---------------- 2. raw feature-dependency plots ----------------
    n_feat = len(features)
    ncols = min(3, n_feat)
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, feat in enumerate(features):
        ax = axes[i // ncols][i % ncols]
        ax.scatter(X_all[:, i], y_all, alpha=0.5, s=15)
        ax.set_xlabel(feat)
        ax.set_ylabel(args.target)
        ax.set_title(f"{feat} vs {args.target} (raw)")
    for j in range(n_feat, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Step 2: raw dependency of each feature on the target")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "01_feature_dependency_raw.png"), dpi=140)
    plt.close(fig)

    # ---------------- 3. term-importance / dependency check, BEFORE picking L1 vs L2 ----------------
    # Expand to the richest polynomial (max_degree), standardise, and look at each term's raw
    # correlation with the target. Purely descriptive -- no model fit yet. This is the piece that
    # tells you whether the effect looks concentrated in a few terms (favours L1, which can zero
    # weak ones out) or spread across most of them (favours L2, which shrinks everything together
    # without eliminating any of it).
    Xrich, poly_rich, scaler_rich = build_poly(X_all, args.max_degree, fit_scaler=True)
    rich_names = term_names(features, args.max_degree)
    corr = np.array([np.corrcoef(Xrich[:, j], y_all)[0, 1] for j in range(Xrich.shape[1])])
    corr = np.nan_to_num(corr)
    order = np.argsort(-np.abs(corr))

    frac_weak = float(np.mean(np.abs(corr) < 0.05))
    print(f"\nTerm-importance check at degree={args.max_degree} (before any regularisation):")
    print(f"  {frac_weak * 100:.0f}% of the {len(corr)} polynomial terms have |corr with y| < 0.05")
    if frac_weak > 0.5:
        print("  -> effect looks CONCENTRATED in a few terms: expect L1 to help (it can zero out the rest).")
    else:
        print("  -> effect looks SPREAD across many terms: expect L2 to help more than L1.")
    print("  (this is only a hint from raw correlation -- the search below decides for real using validation MSE)")

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(corr)), 5))
    ax.bar(range(len(corr)), np.abs(corr[order]), color="teal")
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels([str(rich_names[i]) for i in order], rotation=90, fontsize=6)
    ax.set_ylabel("|correlation with target|")
    ax.set_title(f"Step 3: which terms actually matter, before regularisation (degree={args.max_degree})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "02_term_importance_dependency.png"), dpi=140)
    plt.close(fig)

    # ---------------- 4. JOINT search over degree AND regularisation ----------------
    # Every candidate is evaluated with K-fold CV. Polynomial expansion and scaling are
    # fitted INSIDE each fold using only that fold's training rows, so there is no leakage.
    degree_grid = [args.degree] if args.degree is not None else list(range(1, args.max_degree + 1))
    lambdas = np.logspace(-4, 1.5, args.n_lambdas)

    degree_curves = {}
    train_mse_by_deg, unreg_val_mse_by_deg = [], []
    grid_records = []

    def cv_candidate(degree, reg_type, lam):
        fold_train_mse, fold_val_mse, first_fold_w = [], [], None
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all), start=1):
            Xtr_raw, Xval_raw = X_all[train_idx], X_all[val_idx]
            ytr, yval = y_all[train_idx], y_all[val_idx]

            Xtr_p, poly_fold, scaler_fold = build_poly(Xtr_raw, degree, fit_scaler=True)
            Xval_p = scaler_fold.transform(poly_fold.transform(Xval_raw))

            # No fold-validation data is passed into the optimiser. This prevents the
            # CV validation fold from also controlling early stopping.
            model, _, _ = train_linear_pytorch(
                Xtr_p, ytr, None, None,
                reg_type=reg_type, lam=lam,
                epochs=args.epochs, lr=args.lr, patience=args.epochs)

            tr_mse, _, _ = evaluate(model, Xtr_p, ytr)
            val_mse, _, _ = evaluate(model, Xval_p, yval)
            fold_train_mse.append(tr_mse)
            fold_val_mse.append(val_mse)
            if first_fold_w is None:
                first_fold_w = model.linear.weight.detach().cpu().numpy().ravel().copy()

        return float(np.mean(fold_train_mse)), float(np.mean(fold_val_mse)), first_fold_w

    print("\nJoint search over degree x {none, L1, L2} x lambda using "
          f"{args.n_folds}-fold CV...")
    for d in degree_grid:
        tr_mse, none_val_mse, _ = cv_candidate(d, "none", 0.0)
        train_mse_by_deg.append(tr_mse)
        unreg_val_mse_by_deg.append(none_val_mse)
        grid_records.append((d, "none", 0.0, none_val_mse))

        curves = {"l1": [], "l2": [], "none": none_val_mse}
        for reg_type in ["l1", "l2"]:
            for lam in lambdas:
                _, val_mse, w = cv_candidate(d, reg_type, lam)
                curves[reg_type].append((lam, val_mse, w))
                grid_records.append((d, reg_type, lam, val_mse))
        degree_curves[d] = curves

        best_l1 = min(curves["l1"], key=lambda t: t[1])
        best_l2 = min(curves["l2"], key=lambda t: t[1])
        print(f"  degree {d:2d} | mean CV no-reg MSE={none_val_mse:9.4f} | "
              f"best L1 CV MSE={best_l1[1]:9.4f} (lam={best_l1[0]:.4g}) | "
              f"best L2 CV MSE={best_l2[1]:9.4f} (lam={best_l2[0]:.4g})")

    best_degree, best_reg_type, best_lam, best_val_mse = min(grid_records, key=lambda r: r[3])
    print(f"\n-> Global best from {args.n_folds}-fold CV: degree={best_degree}, "
          f"reg_type={best_reg_type}, lambda={best_lam:.5g}, mean CV MSE={best_val_mse:.4f}")

    # -- plot: unregularised train/val MSE vs degree, just for context --
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(degree_grid, train_mse_by_deg, "o-", label="train MSE (no reg)")
    ax.plot(degree_grid, unreg_val_mse_by_deg, "o-", label="validation MSE (no reg)")
    ax.axvline(best_degree, color="grey", ls="--", lw=1, label=f"final chosen degree={best_degree}")
    ax.set_xlabel("polynomial degree (order)")
    ax.set_ylabel("MSE")
    ax.set_title("Step 4a: degree vs. loss, WITHOUT regularisation (context only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "03_degree_vs_loss_unregularized.png"), dpi=140)
    plt.close(fig)

    # -- plot: the one that actually matters -- best-achievable mean CV MSE per degree, none vs L1 vs L2 --
    best_l1_by_deg = [min(degree_curves[d]["l1"], key=lambda t: t[1])[1] for d in degree_grid]
    best_l2_by_deg = [min(degree_curves[d]["l2"], key=lambda t: t[1])[1] for d in degree_grid]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(degree_grid, unreg_val_mse_by_deg, "o-", label="no regularisation")
    ax.plot(degree_grid, best_l1_by_deg, "o-", label="best L1 at this degree")
    ax.plot(degree_grid, best_l2_by_deg, "o-", label="best L2 at this degree")
    ax.axvline(best_degree, color="grey", ls="--", lw=1, label=f"final chosen degree={best_degree}")
    ax.set_xlabel("polynomial degree (order)")
    ax.set_ylabel("mean CV MSE")
    ax.set_title("Step 4b: degree AND regularisation chosen together")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "04_degree_vs_loss_with_regularisation.png"), dpi=140)
    plt.close(fig)

    # -- at the globally winning degree: L1 vs L2 val loss across lambda, and coefficient paths --
    names = term_names(features, best_degree)
    curves = degree_curves[best_degree]
    l1_lams = [t[0] for t in curves["l1"]]
    l1_mses = [t[1] for t in curves["l1"]]
    l2_mses = [t[1] for t in curves["l2"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(l1_lams, l1_mses, "o-", label="L1 (Lasso-style)")
    ax.plot(l1_lams, l2_mses, "o-", label="L2 (Ridge-style)")
    ax.axhline(curves["none"], color="grey", ls="--", lw=1, label="no regularisation")
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel("mean CV MSE")
    ax.set_title(f"Step 4c: L1 vs L2 mean CV loss (degree={best_degree})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "05_l1_vs_l2_loss.png"), dpi=140)
    plt.close(fig)

    for reg_type, fname, label in [("l1", "06_coef_path_l1.png", "L1"),
                                    ("l2", "07_coef_path_l2.png", "L2")]:
        coefs = np.array([t[2] for t in curves[reg_type]])   # (n_lambdas, n_terms)
        lams = [t[0] for t in curves[reg_type]]
        fig, ax = plt.subplots(figsize=(8, 5))
        for j, name in enumerate(names):
            ax.plot(lams, coefs[:, j], label=str(name))
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel("coefficient value")
        ax.set_title(f"Step 4d: {label} coefficient path (degree={best_degree})")
        if len(names) <= 20:
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, fname), dpi=140)
        plt.close(fig)

    # ---------------- 5. refit on FULL training data with the winning config ----------------
    Xfull_p, poly_final, scaler_final = build_poly(X_all, best_degree, fit_scaler=True)
    final_model, final_train_hist, _ = train_linear_pytorch(
        Xfull_p, y_all, None, None,
        reg_type=best_reg_type, lam=best_lam,
        epochs=args.epochs, lr=args.lr, patience=args.epochs)  # no early stop, no val here

    final_mse, final_r2, final_pred = evaluate(final_model, Xfull_p, y_all)
    print(f"\nFinal model trained on ALL training rows -> train MSE={final_mse:.4f}, "
          f"train R2={final_r2:.4f}")

    # ---------------- 6. model-effect (partial dependence) plots ----------------
    means = X_all.mean(axis=0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, feat in enumerate(features):
        ax = axes[i // ncols][i % ncols]
        grid = np.linspace(X_all[:, i].min(), X_all[:, i].max(), 100)
        X_vary = np.tile(means, (100, 1))
        X_vary[:, i] = grid
        X_vary_p = scaler_final.transform(poly_final.transform(X_vary))
        with torch.no_grad():
            y_vary = final_model(torch.tensor(X_vary_p, dtype=torch.float32)).numpy().ravel()
        ax.plot(grid, y_vary, color="crimson", lw=2)
        ax.scatter(X_all[:, i], y_all, alpha=0.15, s=10, color="steelblue")
        ax.set_xlabel(feat)
        ax.set_ylabel(args.target)
        ax.set_title(f"model effect of {feat}\n(others held at mean)")
    for j in range(n_feat, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Step 6: each feature's effect according to the fitted model")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "08_model_effect_per_feature.png"), dpi=140)
    plt.close(fig)

    # -- final diagnostics: predicted vs actual, residuals --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(y_all, final_pred, alpha=0.4, s=15)
    lims = [min(y_all.min(), final_pred.min()), max(y_all.max(), final_pred.max())]
    ax1.plot(lims, lims, "k--", lw=1)
    ax1.set_xlabel("actual y")
    ax1.set_ylabel("predicted y")
    ax1.set_title(f"Predicted vs actual (R2={final_r2:.3f})")

    residuals = y_all - final_pred
    ax2.scatter(final_pred, residuals, alpha=0.4, s=15)
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.set_xlabel("predicted y")
    ax2.set_ylabel("residual (actual - predicted)")
    ax2.set_title(f"Residuals (train MSE={final_mse:.3f})")
    fig.suptitle("Step 6: final model diagnostics")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "09_final_diagnostics.png"), dpi=140)
    plt.close(fig)

    # ---------------- 7. save everything the prediction script needs ----------------
    config = {
        "features": features,
        "target": args.target,
        "degree": best_degree,
        "reg_type": best_reg_type,
        "lambda": best_lam,
        "poly_input_dim": len(features),
        "scaler_mean": scaler_final.mean_.tolist(),
        "scaler_scale": scaler_final.scale_.tolist(),
        "weight": final_model.linear.weight.detach().numpy().ravel().tolist(),
        "bias": float(final_model.linear.bias.detach().numpy().ravel()[0]),
        "train_mse": final_mse,
        "train_r2": final_r2,
        "term_names": [str(n) for n in names],
    }
    config_path = os.path.join(args.out_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved plots + model_config.json to: {os.path.abspath(args.out_dir)}")
    print("Pass model_config.json to predict_poly_regression.py to generate test predictions.")


if __name__ == "__main__":
    main()
