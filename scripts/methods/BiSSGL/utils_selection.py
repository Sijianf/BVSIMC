"""Feature-selection analysis for the real-data (tuberculosis) experiments.

Companion to utils_real.py, which loads and summarizes the CV results and
compares predictive performance. This module takes the `collected` dict
produced by utils_real.collect()/summarize_method() and analyzes WHICH
features are selected:

  * get_feature_frequency        stability across outer folds at the CV-selected HP
  * get_feature_frequency_grid   Komodromos-style proportion pooled over the HP grid
  * get_feature_frequency_by_hp  the same, stratified by one hyperparameter
  * feature_selection_table      LaTeX table with both proportions side by side
  * fig_sel_heatmap              feature x fold norm heatmap
  * load_feature_names, feature_selection_report, ...  named tables for the
                                 drug / strain feature CSVs + cross-method view

Norm handling: coefficient-row L2 norms pooled across the HP grid are a
mixture of scales (weak penalty -> large norms), so tie-breaking and display
default to the median, and `relative=True` normalizes within each fit.

Typical use
-----------
    import utils_real as ur, utils_selection as us
    collected = ur.collect(OUT_DIR, n_orig_U=n_orig_U, n_orig_V=33)
    names_U, _, prev_U = us.load_feature_names(STRAIN_CSV, id_col="Strain",
                                               n_expected=n_orig_U)      # U = strains
    names_V, _, prev_V = us.load_feature_names(DRUG_CSV, n_skip=4,
                                               n_expected=33)            # V = drugs
    rep = us.feature_selection_report(collected, names_U, names_V, n_orig_U, 33,
                                      prevalence_U=prev_U, prevalence_V=prev_V,
                                      top_n=15, savedir=TAB_DIR)
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_real import (_fmt, _booktabs, _label_for, _style_axis, RC, _INK,
                        _MUTED, _GRID, METHOD_LABEL, get_method_label)


# ====== feature selection frequency analysis (stability across folds) ======
def _sel_items(sel, n_orig, relative=False):
    """Normalize one selection entry to a list of (idx, norm) pairs.

    relative=True divides each norm by the largest selected norm *within
    this same fit* (after n_orig filtering), so values lie in (0, 1] and mean
    'importance relative to the strongest feature in that model'. This makes
    norms comparable across fits at different penalty levels, where raw
    magnitudes differ by orders of magnitude (weak penalty -> huge norms).
    """
    if isinstance(sel, dict):
        items = [(int(k), float(v)) for k, v in sel.items()]
    else:  # backward compatibility: plain list of indices
        items = [(int(k), np.nan) for k in sel]
    if n_orig is not None:
        items = [(i, v) for i, v in items if i < n_orig]
    if relative and items:
        mx = np.nanmax([v for _, v in items])
        if np.isfinite(mx) and mx > 0:
            items = [(i, v / mx) for i, v in items]
    return items


NORM_STATS = ("mean", "median", "min", "max")


def _norm_summary(norms):
    """Aggregate a feature's norms over the fits in which it was selected.
    Returns dict with mean/std, median/iqr, min, max (NaN-safe)."""
    a = np.asarray(norms, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(norm_mean=np.nan, norm_std=np.nan, norm_median=np.nan,
                    norm_iqr=np.nan, norm_min=np.nan, norm_max=np.nan)
    q25, q75 = np.percentile(a, [25, 75])
    return dict(
        norm_mean=float(a.mean()),
        norm_std=float(a.std(ddof=1)) if a.size > 1 else 0.0,
        norm_median=float(np.median(a)),
        norm_iqr=float(q75 - q25),
        norm_min=float(a.min()),
        norm_max=float(a.max()),
    )


def _norm_cols(sort_norm):
    """(center_col, spread_col) used for display of a given statistic."""
    if sort_norm not in NORM_STATS:
        raise ValueError(f"sort_norm must be one of {NORM_STATS}")
    spread = {"mean": "norm_std", "median": "norm_iqr"}.get(sort_norm)
    return f"norm_{sort_norm}", spread


def _build_freq_df(norms_by_idx, n_fits, feature_names, sort_norm):
    """Shared assembly of a frequency DataFrame from {idx: [norms]}."""
    rows = []
    for feat_idx in sorted(norms_by_idx):
        norms = norms_by_idx[feat_idx]
        row = {"feature_idx": feat_idx, "count": len(norms),
               "frequency": len(norms) / n_fits}
        row.update(_norm_summary(norms))
        if feature_names is not None:
            row["feature_name"] = feature_names[feat_idx]
        rows.append(row)
    freq_df = pd.DataFrame(rows)
    if len(freq_df) == 0:
        return freq_df
    col_order = ["feature_idx"]
    if feature_names is not None:
        col_order.append("feature_name")
    col_order += ["count", "frequency", "norm_mean", "norm_std",
                  "norm_median", "norm_iqr", "norm_min", "norm_max"]
    center, _ = _norm_cols(sort_norm)
    # sort by frequency, breaking ties by the chosen norm statistic
    return (freq_df[col_order]
            .sort_values(["count", center], ascending=[False, False])
            .reset_index(drop=True))


def get_feature_frequency(results, sel_col, feature_names=None, n_orig=None,
                          sort_norm="median", relative=False):
    """Count selection frequency and aggregate importance (L2 norm) across
    outer folds at the CV-selected hyperparameters. `sel_col` holds dicts of
    {feature_idx: L2_row_norm}: test_sel_A / test_sel_W = U (row, strain)
    side, test_sel_B / test_sel_H = V (column, drug) side.

    n_orig    : exclude identity-augmented indices (idx >= n_orig)
    sort_norm : which norm statistic breaks frequency ties, one of
                'mean', 'median' (default, robust to a few huge norms),
                'min', 'max'
    relative  : normalize norms within each fit by that fit's largest
                selected norm before aggregating (see _sel_items)

    Returns a DataFrame sorted by count desc (ties by `sort_norm`) with
    columns: feature_idx, [feature_name], count, frequency,
    norm_mean, norm_std, norm_median, norm_iqr, norm_min, norm_max.
    """
    tpf = results["test_per_fold"]
    if sel_col not in tpf.columns:
        available = [c for c in tpf.columns if c.startswith("test_sel_")]
        raise KeyError(f"Column '{sel_col}' not found in test_per_fold. "
                       f"Available selection columns: {available}")

    n_folds = len(tpf)
    norms_by_idx = defaultdict(list)
    for sel in tpf[sel_col]:
        for idx, norm in _sel_items(sel, n_orig, relative=relative):
            norms_by_idx[idx].append(norm)
    return _build_freq_df(norms_by_idx, n_folds, feature_names, sort_norm)


def sel_norm_matrix(results, sel_col, feature_names=None, n_orig=None,
                    fill=np.nan):
    """Expand a selection column into a (outer fold x feature) norm matrix.

    Unlike score vectors, the test_sel_* columns hold ragged dicts
    {feature_idx: L2_row_norm} whose keys differ per fold, so expand_scores
    (which needs fixed-length, fixed-order vectors) cannot be used. This
    unions the keys instead: one row per (trial, fold), one column per
    feature ever selected, `fill` (default NaN) where a feature was not
    selected in that fold. Use fill=0.0 to treat non-selection as zero norm.

    Returns a DataFrame indexed by (trial, fold), columns sorted by
    feature index (renamed via feature_names if given).
    """
    tpf = results["test_per_fold"]
    if sel_col not in tpf.columns:
        available = [c for c in tpf.columns if c.startswith("test_sel_")]
        raise KeyError(f"Column '{sel_col}' not found in test_per_fold. "
                       f"Available selection columns: {available}")

    records = []
    for sel in tpf[sel_col]:
        if isinstance(sel, dict):
            rec = {int(k): float(v) for k, v in sel.items()}
        else:  # backward compatibility: plain list of indices
            rec = {int(k): 1.0 for k in sel}
        if n_orig is not None:
            rec = {k: v for k, v in rec.items() if k < n_orig}
        records.append(rec)

    mat = pd.DataFrame(records,
                       index=pd.MultiIndex.from_frame(tpf[["trial", "fold"]]))
    mat = mat[sorted(mat.columns)]
    if not np.isnan(fill):
        mat = mat.fillna(fill)
    if feature_names is not None:
        mat.columns = [feature_names[i] for i in mat.columns]
    return mat


# ====== grid-pooled selection frequency (Komodromos et al. 2022 style) ======
def _infer_hyperparam_cols(results):
    """Recover the hyperparameter columns from results['best_params']
    (everything that is not an id/diagnostic/score column)."""
    skip = {"trial", "fold", "n_inner_cv"}
    return [c for c in results["best_params"].columns
            if c not in skip and not c.startswith("val_score")]


def _fits_frame(results, sel_col, hyperparam_cols=None):
    """One row per fit = (trial, fold, HP combo), with its test selection
    dict. test_sel_* is identical across inner CV folds for the same combo
    (like test_score), so the first inner-CV row is taken per fit."""
    raw = results["raw"]
    if sel_col not in raw.columns:
        available = [c for c in raw.columns if c.startswith("test_sel_")]
        raise KeyError(f"Column '{sel_col}' not found in raw results. "
                       f"Available selection columns: {available}")
    hyperparam_cols = hyperparam_cols or _infer_hyperparam_cols(results)
    fit_cols = ["trial", "fold"] + hyperparam_cols
    fits = (raw.groupby(fit_cols, as_index=False, sort=False)
               .agg(**{sel_col: (sel_col, "first")}))
    return fits, hyperparam_cols


def get_feature_frequency_grid(results, sel_col, hyperparam_cols=None,
                               feature_names=None, n_orig=None,
                               sort_norm="median", relative=False):
    """Selection frequency pooled over the full hyperparameter grid AND all
    outer folds/trials, mirroring the 'selection proportion' of Komodromos
    et al. (2022, Bioinformatics 38:3918), whose denominator was
    (14 lambda values x 10 folds) = 140 fits.

    Here: frequency = (# fits selecting the feature) / n_fits, where a fit
    is one (trial, outer fold, HP combo) and n_fits = trials x folds x grid.

    NORMS ACROSS THE GRID ARE A MIXTURE OF SCALES: weak-penalty fits give
    much larger coefficients than strong-penalty fits, so the raw mean is
    dominated by the weakest penalty. Hence sort_norm defaults to 'median';
    relative=True (per-fit normalization) is the cleaner fix and is
    recommended for tie-breaking / display in the pooled setting.

    CAVEAT: pooling over the grid is only meaningful if selection is not
    too sensitive to the hyperparameters (as Komodromos et al. verified for
    lambda). Pair with get_feature_frequency() (CV-selected HP only) and
    get_feature_frequency_by_hp() (stratified) when reporting.

    Returns a DataFrame sorted by count desc (ties by `sort_norm`) with
    columns: feature_idx, [feature_name], count, frequency,
    norm_mean, norm_std, norm_median, norm_iqr, norm_min, norm_max.
    """
    fits, _ = _fits_frame(results, sel_col, hyperparam_cols)
    n_fits = len(fits)
    norms_by_idx = defaultdict(list)
    for sel in fits[sel_col]:
        for idx, norm in _sel_items(sel, n_orig, relative=relative):
            norms_by_idx[idx].append(norm)
    return _build_freq_df(norms_by_idx, n_fits, feature_names, sort_norm)


def get_feature_frequency_by_hp(results, sel_col, stratify_by,
                                hyperparam_cols=None, feature_names=None,
                                n_orig=None, top_n=None, include_norm=False,
                                sort_norm="median", relative=False):
    """Selection frequency stratified by one hyperparameter: one column per
    value of `stratify_by` (e.g. 'tilde_lambda0'), frequency computed over
    the fits (trials x folds x remaining grid) within that stratum.

    This is the honest companion to get_feature_frequency_grid() when
    selection is hyperparameter-sensitive: consistent rows across columns
    support a stability claim; rows that light up only at small penalties
    reveal grid dependence that pooling would average away.

    n_orig, if given, excludes identity-augmented indices (idx >= n_orig)
    from counts and denominators, as in the other frequency functions.

    Returns a DataFrame indexed by feature (name if feature_names given,
    else index), columns = sorted values of `stratify_by`, plus a 'pooled'
    column (and 'norm_<sort_norm>' if include_norm=True); rows sorted by
    pooled frequency desc with ties broken by the `sort_norm` statistic of
    the (optionally per-fit `relative`) norm when selected -- the same order
    as get_feature_frequency_grid() with the same options -- truncated to
    top_n.
    """
    fits, hyperparam_cols = _fits_frame(results, sel_col, hyperparam_cols)
    if stratify_by not in hyperparam_cols:
        raise KeyError(f"'{stratify_by}' is not a hyperparameter column "
                       f"({hyperparam_cols})")

    counts = defaultdict(lambda: defaultdict(int))   # idx -> stratum -> count
    norms_by_idx = defaultdict(list)                 # idx -> pooled norms
    n_by_stratum = defaultdict(int)
    for stratum, sel in zip(fits[stratify_by], fits[sel_col]):
        n_by_stratum[stratum] += 1
        for idx, norm in _sel_items(sel, n_orig, relative=relative):
            counts[idx][stratum] += 1
            norms_by_idx[idx].append(norm)

    center, _ = _norm_cols(sort_norm)
    strata = sorted(n_by_stratum)
    n_fits = sum(n_by_stratum.values())
    rows, index = [], []
    for feat_idx in sorted(counts):
        by_s = counts[feat_idx]
        row = {s: by_s.get(s, 0) / n_by_stratum[s] for s in strata}
        row["pooled"] = sum(by_s.values()) / n_fits
        row[center] = _norm_summary(norms_by_idx[feat_idx])[center]
        rows.append(row)
        index.append(feature_names[feat_idx] if feature_names is not None
                     else feat_idx)

    out = pd.DataFrame(rows, index=index, columns=strata + ["pooled", center])
    out.index.name = "feature"
    out = out.sort_values(["pooled", center], ascending=[False, False])
    if not include_norm:
        out = out.drop(columns=[center])
    return out.head(top_n) if top_n else out


def feature_selection_table(results, sel_col, hyperparam_cols=None,
                            feature_names=None, n_orig=None, top_n=15,
                            prec=2, side_label="row-side features",
                            sort_norm="median", relative=False,
                            caption=None, label=None, savepath=None):
    """Publication table of the most frequently selected features, reporting
    BOTH selection proportions side by side:

      - Freq. (grid):        pooled over trials x folds x full HP grid
                             (Komodromos et al. 2022 style; rank order)
      - Freq. (selected HP): across outer folds at the CV-selected HP combo
                             only (pure data-resampling stability)
      - ||row||_2:           `sort_norm` statistic of the coefficient-row
                             norm over the fits in which the feature was
                             selected (grid pooling): 'median (IQR)' by
                             default, 'mean (sd)', or a bare 'min'/'max'.
                             relative=True reports the per-fit relative norm
                             in (0, 1] instead of the raw magnitude.

    Rows are ordered by grid frequency, ties broken by the same statistic.
    Returns (display_df, latex_str); writes the .tex if savepath is given.
    """
    grid = get_feature_frequency_grid(results, sel_col, hyperparam_cols,
                                      feature_names=feature_names,
                                      n_orig=n_orig, sort_norm=sort_norm,
                                      relative=relative)
    sel = get_feature_frequency(results, sel_col,
                                feature_names=feature_names, n_orig=n_orig,
                                sort_norm=sort_norm, relative=relative)
    center, spread = _norm_cols(sort_norm)
    if len(grid) == 0:
        raise ValueError(f"No features ever selected in '{sel_col}'.")

    sel_freq = dict(zip(sel["feature_idx"], sel["frequency"])) if len(sel) \
        else {}
    n_fits = int(round(grid["count"].iloc[0] / grid["frequency"].iloc[0]))
    n_outer = len(results["test_per_fold"])

    top = grid.head(top_n)
    name_col = "feature_name" if feature_names is not None else "feature_idx"

    disp = pd.DataFrame({
        "Feature": top[name_col].values,
        "Freq. (grid)": top["frequency"].round(prec + 1).values,
        "Freq. (selected HP)": [round(sel_freq.get(i, 0.0), prec + 1)
                                for i in top["feature_idx"]],
        "Norm": [_fmt(m, (s if spread else None), prec) for m, s in
                 zip(top[center], top[spread] if spread else top[center])],
    })

    norm_head = ("$\\|A_{j\\cdot}\\|_2 / \\max_j\\|A_{j\\cdot}\\|_2$"
                 if relative else "$\\|A_{j\\cdot}\\|_2$")
    header = f"Feature & Freq. (grid) & Freq. (sel. HP) & {norm_head} \\\\"
    body = []
    for _, r in disp.iterrows():
        feat_tex = str(r["Feature"]).replace("_", "\\_")
        body.append(f"{feat_tex} & {r['Freq. (grid)']:.{prec}f} & "
                    f"{r['Freq. (selected HP)']:.{prec}f} & {r['Norm']} \\\\")
    norm_desc = {"mean": "mean (sd)", "median": "median (IQR)",
                 "min": "minimum", "max": "maximum"}[sort_norm]
    rel_desc = (", relative to the largest selected norm within each fit,"
                if relative else "")
    caption = caption or (
        f"Most frequently selected {side_label}. Freq.\\ (grid): proportion "
        f"of the {n_fits} fits (trials $\\times$ outer folds $\\times$ "
        f"hyperparameter grid) selecting the feature; Freq.\\ (sel.\\ HP): "
        f"proportion of the {n_outer} outer folds at the CV-selected "
        f"hyperparameters; norm is the {norm_desc} coefficient-row "
        f"$\\ell_2$ norm{rel_desc} over the fits in which the feature was "
        f"selected."
    )
    latex = _booktabs([header], body, "lccc", caption,
                      label or "tab:selection_freq")
    if savepath:
        with open(savepath, "w") as f:
            f.write(latex + "\n")
    return disp, latex


def print_top_features(freq_df, top_n=20, label="Features"):
    """Print the top-N most frequently selected features with importance."""
    if len(freq_df) == 0:
        print(f"No selected features to report for {label}.")
        return
    n_total = int(round(freq_df["count"].iloc[0] / freq_df["frequency"].iloc[0]))
    print(f"\n{'=' * 70}")
    print(f"  Top {top_n} {label}  (out of {n_total} fits)")
    print(f"{'=' * 70}")
    has_name = "feature_name" in freq_df.columns
    has_norm = ("norm_median" in freq_df.columns
                and not freq_df["norm_median"].isna().all())
    for _, row in freq_df.head(top_n).iterrows():
        name_str = f"  {row['feature_name']}" if has_name else ""
        norm_str = (f"  ||r|| med={row['norm_median']:.3f} "
                    f"[{row['norm_min']:.3f}, {row['norm_max']:.3f}]"
                    if has_norm else "")
        # note: iterrows() casts all-numeric rows to float, so re-int here
        print(f"  idx {int(row['feature_idx']):>5d}{name_str:30s}  "
              f"count={int(row['count']):>3d}  "
              f"freq={row['frequency']:.0%}{norm_str}")
    print(f"{'=' * 70}\n")





# ====== figure: selection-stability heatmap (outer folds x top features) ======
def fig_sel_heatmap(results, sel_col, top_n=30, feature_names=None, n_orig=None,
                    method_label=None, cmap="Oranges", orient="horizontal",
                    figsize=None, savepath=None):
    """Heatmap of coefficient-row L2 norms across outer folds, restricted to
    the top_n features by selection frequency (ties broken by mean norm — same
    order as print_top_features). Empty cells = not selected in that fold.
    Thin lines separate trials.

    orient : "horizontal" -> folds on y, features on x (wide; good for few
                             features, e.g. the drug/column (V) side)
             "vertical"   -> features on y, folds on x (tall; good for many
                             features, e.g. the strain/row (U) side)

    Example
    -------
    fig_sel_heatmap(collected["bissgl"], "test_sel_A", top_n=40,
                    orient="vertical", method_label="BVSIMC, strain side")
    """
    if orient not in ("horizontal", "vertical"):
        raise ValueError("orient must be 'horizontal' or 'vertical'")

    freq = get_feature_frequency(results, sel_col, feature_names=feature_names,
                                 n_orig=n_orig)
    if len(freq) == 0:
        return None
    name_col = "feature_name" if feature_names is not None else "feature_idx"
    top = freq.head(top_n)
    mat = sel_norm_matrix(results, sel_col, feature_names=feature_names,
                          n_orig=n_orig)
    mat = mat[top[name_col].tolist()]                # order = freq, then norm

    n_folds, n_feat = mat.shape
    trials = mat.index.get_level_values("trial").values
    bounds = np.flatnonzero(np.diff(trials)) + 1     # trial boundaries
    edges = np.r_[0, bounds, n_folds]
    trial_starts = np.r_[0, bounds].astype(int)
    fold_ticks = (edges[:-1] + edges[1:]) / 2
    fold_labels = [f"trial {t}" for t in trials[trial_starts]]

    # data grid is (fold, feature); transpose for the vertical layout
    grid = mat.values if orient == "horizontal" else mat.values.T
    masked = np.ma.masked_invalid(grid)
    cbar_label = r"$\|A_{j\cdot}\|_2$ (coefficient-row norm)"
    title = "Selected features across outer folds"
    if method_label:
        title += f" — {method_label}"
    else:
        title = None

    with plt.rc_context(RC):
        if orient == "horizontal":
            figsize = figsize or (max(6.0, 0.22 * n_feat + 1.8),
                                  max(3.6, 0.09 * n_folds + 1.6))
        else:
            figsize = figsize or (max(4.5, 0.10 * n_folds + 1.8),
                                  max(4.0, 0.20 * n_feat + 1.6))
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(masked, cmap=cmap, edgecolors="white",
                           linewidth=0.4, vmin=0)

        if orient == "horizontal":
            ax.invert_yaxis()                        # fold 0 at the top
            for b in bounds:
                ax.axhline(b, color=_INK, linewidth=0.9)
            ax.set_yticks(fold_ticks)
            ax.set_yticklabels(fold_labels, fontsize=9)
            ax.set_xticks(np.arange(n_feat) + 0.5)
            ax.set_xticklabels(mat.columns, rotation=90, fontsize=8)
        else:                                        # vertical
            ax.invert_yaxis()                        # top feature at the top
            for b in bounds:
                ax.axvline(b, color=_INK, linewidth=0.9)
            ax.set_xticks(fold_ticks)
            ax.set_xticklabels(fold_labels, fontsize=9, rotation=90)
            ax.set_yticks(np.arange(n_feat) + 0.5)
            ax.set_yticklabels(mat.columns, fontsize=8)

        ax.tick_params(length=0, colors=_MUTED, labelcolor=_INK)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=12, pad=10, color=_INK, loc="left")

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(cbar_label, fontsize=10)
        cbar.outline.set_visible(False)

        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight",
                        facecolor="white")
        return fig


# =============================================================================
# Named feature-selection summaries (real-data analysis)
# =============================================================================
# Workflow:
#   names_U, strains, df_str = load_feature_names(STRAIN_CSV, id_col=...)   # U = strains
#   names_V, drugs,   df_drug = load_feature_names(DRUG_CSV, n_skip=4)      # V = drugs
#   tabs = feature_selection_report(collected, names_U, names_V,
#                                   n_orig_U=len(names_U), n_orig_V=len(names_V),
#                                   prevalence_U=df_str, prevalence_V=df_drug,
#                                   savedir=TAB_DIR)
#
# Index convention: feature index j in test_sel_* refers to the j-th column of
# the (un-augmented) U (row side, strains/diseases) or V (column side, drugs)
# matrix, which must be the j-th feature column of the corresponding CSV. Identity-
# augmented columns (j >= n_orig) are excluded everywhere via n_orig.

_SIDE_SEL_COLS = {
    "row": ["test_sel_A", "test_sel_W"],   # U side: strains/diseases (BiSSGL A / SGIMC W)
    "col": ["test_sel_B", "test_sel_H"],   # V side: drugs            (BiSSGL B / SGIMC H)
}
_SIDE_ALIASES = {"row": "row", "u": "row", "a": "row", "w": "row",
                 "strain": "row", "disease": "row",
                 "col": "col", "v": "col", "b": "col", "h": "col", "drug": "col"}


def _side_sel_col(results, side):
    """Resolve the selection column for a side ('row'/'U'/'A' or 'col'/'V'/'B')
    in a summarize result dict, whichever method produced it."""
    key = _SIDE_ALIASES[str(side).lower()]
    tpf = results["test_per_fold"]
    for c in _SIDE_SEL_COLS[key]:
        if c in tpf.columns:
            return c
    raise KeyError(f"No selection column for side '{side}' in results "
                   f"(looked for {_SIDE_SEL_COLS[key]})")


def load_feature_names(csv_path, id_col=None, n_skip=None, n_expected=None,
                       verbose=True):
    """Read a feature CSV and return (feature_names, entity_ids, feature_df).

    feature_names : list[str]   column names of the feature block, in file
                                order -- must match the column order of the
                                U / V matrix used for fitting
    entity_ids    : list | None values of `id_col` (drug / strain names)
    feature_df    : DataFrame   the feature block only (entities x features),
                                indexed by entity id if id_col is given;
                                useful for prevalence (column sums)

    Feature-block detection, in priority order:
      n_skip   : take all columns from position n_skip onwards (e.g. n_skip=4
                 for drugs_features.csv whose first 4 columns are metadata);
                 id_col may still be given to label the rows (drug names)
      id_col   : drop id_col, then keep only numeric columns
      default  : keep only numeric columns
    If n_expected (= n_orig) is given and the count differs, a warning is
    printed -- fix the alignment before trusting any named table.
    """
    df = pd.read_csv(csv_path)
    if n_skip is not None:
        feat = df.iloc[:, n_skip:]
    else:
        feat = df.drop(columns=[id_col]) if id_col else df
        feat = feat.select_dtypes(include=[np.number])
    names = [str(c) for c in feat.columns]
    ids = df[id_col].astype(str).tolist() if id_col else None
    if ids is not None:
        feat = feat.set_axis(ids, axis=0)
    if n_expected is not None and len(names) != n_expected:
        print(f"WARNING: {os.path.basename(csv_path)} has {len(names)} feature "
              f"columns but n_expected={n_expected}. Column order / count must "
              f"match the fitted U/V matrix; check n_skip/id_col.")
    elif verbose:
        print(f"{os.path.basename(csv_path)}: {len(names)} features"
              + (f", {len(ids)} entities" if ids else ""))
    return names, ids, feat


def pretty_name(name, labels=None, max_len=None):
    """Human-readable feature label: optional override dict, then
    underscores -> spaces; truncate with an ellipsis if max_len is set."""
    s = (labels or {}).get(name, str(name).replace("_", " "))
    if max_len and len(s) > max_len:
        s = s[:max_len - 1] + "\u2026"
    return s


def _tex(s):
    return (str(s).replace("\\", "\\textbackslash ").replace("&", "\\&")
            .replace("%", "\\%").replace("_", "\\_").replace("#", "\\#"))


def _prevalence_counts(feature_df):
    """Number of entities carrying each feature = count of NON-ZERO entries
    per column. Equals the column sum for 0/1 indicators, and stays meaningful
    for -1/+1 or non-binary encodings (where the plain sum would be ~0)."""
    num = feature_df.apply(pd.to_numeric, errors="coerce")
    nonbin = [c for c in num.columns
              if not num[c].dropna().isin([0, 1]).all()]
    if nonbin:
        print(f"NOTE: {len(nonbin)} feature column(s) are not 0/1 (e.g. "
              f"{nonbin[:3]}); prevalence counts non-zero entries.")
    return (num.fillna(0) != 0).sum(axis=0)


def check_feature_alignment(feature_df, M, n_orig=None, atol=1e-8,
                            name="V"):
    """Verify that the CSV feature block equals the fitted matrix (its first
    n_orig columns if identity-augmented). Prints a verdict and returns True
    if aligned. Call with the matrix actually passed to the model, e.g.
        X, Y, R = load(staged_dataset);  check_feature_alignment(prev_V, Y, 33)
    """
    A = np.asarray(M.todense() if hasattr(M, "todense") else M, dtype=float)
    n_orig = n_orig or feature_df.shape[1]
    B = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0).values
    if A.shape[0] != B.shape[0]:
        print(f"[{name}] row mismatch: matrix has {A.shape[0]} rows, CSV has "
              f"{B.shape[0]} entities -> different entity sets/order.")
        return False
    if A.shape[1] < n_orig:
        print(f"[{name}] matrix has only {A.shape[1]} columns < n_orig={n_orig}.")
        return False
    A = A[:, :n_orig]
    if B.shape[1] != n_orig:
        print(f"[{name}] CSV has {B.shape[1]} feature columns, n_orig={n_orig}.")
        return False
    diff = np.abs(A - B) > atol
    if not diff.any():
        print(f"[{name}] aligned: CSV feature block == matrix[:, :{n_orig}].")
        return True
    bad_cols = np.flatnonzero(diff.any(axis=0))
    print(f"[{name}] NOT aligned: {len(bad_cols)}/{n_orig} columns differ "
          f"(first: {bad_cols[:8].tolist()}); names/prevalence would be wrong.")
    # helpful hint: is it a pure column permutation?
    try:
        perm = [next(j for j in range(n_orig) if np.allclose(A[:, i], B[:, j],
                                                              atol=atol))
                for i in range(n_orig)]
        print(f"[{name}] matrix columns map to CSV columns {perm[:10]}... "
              "-> reorder the CSV feature block accordingly.")
    except StopIteration:
        pass
    return False


def feature_table_named(results, side, feature_names, n_orig, top_n=15,
                        prevalence=None, labels=None, sort_norm="median",
                        relative=True, prec=2, hyperparam_cols=None,
                        tie_break="norm", prevalence_range=None):
    """Tidy DataFrame of the top selected features on one side, by name.

    Columns
    -------
    rank, feature (pretty), feature_raw, feature_idx,
    freq_grid     : proportion of (trials x folds x HP grid) fits selecting it
    freq_selhp    : proportion of outer folds at the CV-selected HP
    n_strata_any  : (only if a by-HP view is cheap) omitted here
    norm          : `sort_norm` statistic of the (relative) row norm
    norm_spread   : IQR (median) / sd (mean) / NaN (min, max)
    prevalence    : if `prevalence` (entities x features DataFrame) is given,
                    number of entities with a non-zero value for the feature
                    -- e.g. how many drugs carry the group
    prevalence_pct: the same as a fraction of entities

    Ranking: always by freq_grid (the evidence of consistent selection),
    then ties broken by `tie_break`:
      'norm'            relative row norm (default; the model's effect size)
      'prevalence'      more common features first (needs `prevalence`)
      'prevalence_low'  rarer features first -- specific groups above
                        near-universal ones
    prevalence_range : optional (lo, hi) fraction bounds; features whose
                       prevalence_pct falls outside are dropped BEFORE taking
                       top_n (e.g. (0.05, 0.9) removes near-constant groups).
                       State the filter in the caption if you use it.
    """
    sel_col = _side_sel_col(results, side)
    grid = get_feature_frequency_grid(results, sel_col, hyperparam_cols,
                                      feature_names=feature_names,
                                      n_orig=n_orig, sort_norm=sort_norm,
                                      relative=relative)
    sel = get_feature_frequency(results, sel_col, feature_names=feature_names,
                                n_orig=n_orig, sort_norm=sort_norm,
                                relative=relative)
    if len(grid) == 0:
        return pd.DataFrame()
    center, spread = _norm_cols(sort_norm)
    sel_freq = dict(zip(sel["feature_idx"], sel["frequency"])) if len(sel) else {}

    if (tie_break.startswith("prevalence") or prevalence_range) \
            and prevalence is None:
        raise ValueError("tie_break='prevalence*' / prevalence_range need the "
                         "`prevalence` DataFrame.")
    if prevalence is not None:
        counts_all = _prevalence_counts(prevalence)
        grid["_prev"] = [float(counts_all.get(nm, np.nan))
                         for nm in grid["feature_name"]]
        if prevalence_range is not None:
            lo, hi = prevalence_range
            frac = grid["_prev"] / len(prevalence)
            grid = grid[(frac >= lo) & (frac <= hi)]
        if tie_break == "prevalence":
            grid = grid.sort_values(["count", "_prev", center],
                                    ascending=[False, False, False])
        elif tie_break == "prevalence_low":
            grid = grid.sort_values(["count", "_prev", center],
                                    ascending=[False, True, False])
    top = grid.head(top_n).reset_index(drop=True)
    out = pd.DataFrame({
        "rank": np.arange(1, len(top) + 1),
        "feature": [pretty_name(n, labels) for n in top["feature_name"]],
        "feature_raw": top["feature_name"].values,
        "feature_idx": top["feature_idx"].values,
        "freq_grid": top["frequency"].values,
        "freq_selhp": [sel_freq.get(i, 0.0) for i in top["feature_idx"]],
        "norm": top[center].values,
        "norm_spread": top[spread].values if spread else np.nan,
    })
    if prevalence is not None:
        counts = _prevalence_counts(prevalence)
        out["prevalence"] = [int(counts[c]) if c in counts.index else np.nan
                             for c in top["feature_name"]]
        out["prevalence_pct"] = out["prevalence"] / len(prevalence)
        bad = out[(out["prevalence"] == 0) & (out["freq_grid"] > 0)]
        if len(bad):
            print("WARNING: selected feature(s) with zero prevalence in the CSV: "
                  f"{list(bad['feature_raw'])}. A feature that is all-zero in V "
                  "cannot be selected, so the CSV columns are probably not "
                  "aligned with the fitted matrix (check n_skip / column "
                  "order / file version with check_feature_alignment()).")
    out.attrs["n_fits"] = int(round(grid["count"].iloc[0] /
                                    grid["frequency"].iloc[0]))
    out.attrs["n_outer"] = len(results["test_per_fold"])
    out.attrs["sel_col"] = sel_col
    out.attrs["sort_norm"], out.attrs["relative"] = sort_norm, relative
    return out


def feature_table_named_latex(tab, side_label, caption=None, label=None,
                              prec=2, savepath=None, matrix_symbol=None):
    """Render the DataFrame from feature_table_named() as a booktabs table."""
    if len(tab) == 0:
        raise ValueError("Empty feature table.")
    has_prev = "prevalence" in tab.columns
    sort_norm, relative = tab.attrs["sort_norm"], tab.attrs["relative"]
    spread_ok = sort_norm in ("mean", "median")
    sym = matrix_symbol or "A"
    norm_head = (f"$\\|{sym}_{{j\\cdot}}\\|_2/\\max_j\\|{sym}_{{j\\cdot}}\\|_2$"
                 if relative else f"$\\|{sym}_{{j\\cdot}}\\|_2$")
    head = ["Rank & Feature & Freq.\\ (grid) & Freq.\\ (sel.\\ HP) & "
            + norm_head + (" & Prevalence" if has_prev else "") + " \\\\"]
    body = []
    for _, r in tab.iterrows():
        norm_str = _fmt(r["norm"], r["norm_spread"] if spread_ok else None, prec)
        row = (f"{int(r['rank'])} & {_tex(r['feature'])} & "
               f"{r['freq_grid']:.{prec}f} & {r['freq_selhp']:.{prec}f} & "
               f"{norm_str}")
        if has_prev:
            row += (f" & {int(r['prevalence'])} ({100*r['prevalence_pct']:.0f}\\%)"
                    if np.isfinite(r["prevalence"]) else " & --")
        body.append(row + " \\\\")
    norm_desc = {"mean": "mean (sd)", "median": "median (IQR)",
                 "min": "minimum", "max": "maximum"}[sort_norm]
    caption = caption or (
        f"Most frequently selected {side_label}. Freq.\\ (grid): proportion of "
        f"the {tab.attrs['n_fits']} fits (trials $\\times$ outer folds $\\times$ "
        f"hyperparameter grid) in which the feature is selected; Freq.\\ (sel.\\ "
        f"HP): proportion of the {tab.attrs['n_outer']} outer folds at the "
        f"CV-selected hyperparameters; norm: {norm_desc} coefficient-row "
        f"$\\ell_2$ norm{', relative to the largest selected norm within each fit,' if relative else ''} "
        f"over the fits in which the feature is selected"
        + ("; prevalence: number (\\%) of entities (drugs/strains) whose "
           "feature value is non-zero in the input data."
           if has_prev else ".")
    )
    colfmt = "rlccc" + ("c" if has_prev else "")
    latex = _booktabs(head, body, colfmt, caption,
                      label or f"tab:sel_{side_label.split()[0].lower()}")
    if savepath:
        with open(savepath, "w") as f:
            f.write(latex + "\n")
    return latex


def feature_table_across_methods(collected, side, feature_names, n_orig,
                                 methods=None, top_n=15, labels=None,
                                 rank_by="bissgl", freq="grid",
                                 sort_norm="median", relative=True,
                                 prevalence=None):
    """Side-by-side selection frequency of the top features across methods.

    Rows: union of each method's top-`top_n` features, ordered by `rank_by`'s
    frequency (then by the other methods'). Columns: one per method with the
    chosen frequency ('grid' = pooled over the HP grid, 'selhp' = outer folds
    at the CV-selected HP), plus optional prevalence. Methods lacking
    selection dicts (e.g. drimc/nrlmf) are skipped.
    """
    methods = [m for m in (methods or list(collected))
               if any(c.startswith("test_sel_")
                      for c in collected[m]["test_per_fold"].columns)]
    per_method = {}
    for m in methods:
        try:
            sel_col = _side_sel_col(collected[m], side)
        except KeyError:
            continue
        f = (get_feature_frequency_grid if freq == "grid"
             else get_feature_frequency)
        kw = dict(feature_names=feature_names, n_orig=n_orig,
                  sort_norm=sort_norm, relative=relative)
        df = f(collected[m], sel_col, **kw)
        per_method[m] = df.set_index("feature_idx")
    if not per_method:
        return pd.DataFrame()

    union = []
    for m, df in per_method.items():
        union += list(df.index[:top_n])
    union = list(dict.fromkeys(union))           # preserve first-seen order

    out = pd.DataFrame({"feature_idx": union})
    out["feature"] = [pretty_name(feature_names[i], labels) for i in union]
    for m, df in per_method.items():
        out[_label_for(m, collected)] = [
            float(df["frequency"].get(i, 0.0)) for i in union]
    if prevalence is not None:
        cnt = _prevalence_counts(prevalence)
        out["prevalence"] = [int(cnt.get(feature_names[i], 0)) for i in union]
    order_cols = ([_label_for(rank_by, collected)] if rank_by in per_method
                  else []) + [_label_for(m, collected) for m in per_method
                              if m != rank_by]
    out = out.sort_values(order_cols, ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.attrs["methods"] = [_label_for(m, collected) for m in per_method]
    out.attrs["freq"] = freq
    return out


def feature_table_across_methods_latex(tab, side_label, caption=None,
                                       label=None, prec=2, savepath=None):
    methods = tab.attrs["methods"]
    has_prev = "prevalence" in tab.columns
    head = ["Rank & Feature & " + " & ".join(methods)
            + (" & Prevalence" if has_prev else "") + " \\\\"]
    body = []
    for _, r in tab.iterrows():
        cells = [f"{r[m]:.{prec}f}" for m in methods]
        # bold the best frequency per row
        best = max(r[m] for m in methods)
        cells = [f"\\textbf{{{c}}}" if abs(r[m] - best) < 1e-12 and best > 0
                 else c for c, m in zip(cells, methods)]
        row = f"{int(r['rank'])} & {_tex(r['feature'])} & " + " & ".join(cells)
        if has_prev:
            row += f" & {int(r['prevalence'])}"
        body.append(row + " \\\\")
    what = ("pooled over trials, outer folds and the hyperparameter grid"
            if tab.attrs["freq"] == "grid"
            else "across outer folds at the CV-selected hyperparameters")
    caption = caption or (
        f"Selection frequency of the most frequently selected {side_label} "
        f"by method ({what}); rows are the union of each method's top "
        f"features, best value per row in bold"
        + ("; prevalence: number of entities carrying the feature."
           if has_prev else "."))
    colfmt = "rl" + "c" * len(methods) + ("c" if has_prev else "")
    latex = _booktabs(head, body, colfmt, caption,
                      label or f"tab:sel_methods_{side_label.split()[0].lower()}")
    if savepath:
        with open(savepath, "w") as f:
            f.write(latex + "\n")
    return latex


def feature_selection_report(collected, names_U, names_V, n_orig_U, n_orig_V,
                             method="bissgl", top_n=15, prevalence_U=None,
                             prevalence_V=None, labels_U=None, labels_V=None,
                             side_labels=("strain features",
                                          "drug functional groups"),
                             sort_norm="median", relative=True, prec=2,
                             tie_break="norm", prevalence_range=None,
                             savedir=None, verbose=True):
    """One call: named top-feature tables for both sides of `method`, the
    cross-method comparison for both sides, full-length CSVs, and .tex files.

    Returns dict with keys 'row', 'col' (method tables), 'row_methods',
    'col_methods' (cross-method), each a DataFrame; LaTeX strings under
    the same keys with suffix '_tex'.
    """
    out = {}
    res = collected[method]
    specs = [("row", names_U, n_orig_U, prevalence_U, labels_U, side_labels[0], "A"),
             ("col", names_V, n_orig_V, prevalence_V, labels_V, side_labels[1], "B")]
    if savedir:
        os.makedirs(savedir, exist_ok=True)
    for side, names, n_orig, prev, labels, slabel, sym in specs:
        if len(names) != n_orig:
            print(f"WARNING [{side}]: len(feature_names)={len(names)} != "
                  f"n_orig={n_orig}; names may be misaligned with indices.")
        tab = feature_table_named(res, side, names, n_orig, top_n=top_n,
                                  prevalence=prev, labels=labels,
                                  sort_norm=sort_norm, relative=relative,
                                  tie_break=(tie_break if prev is not None
                                             else "norm"),
                                  prevalence_range=(prevalence_range
                                                    if prev is not None
                                                    else None))
        out[side] = tab
        if len(tab):
            out[side + "_tex"] = feature_table_named_latex(
                tab, slabel, prec=prec, matrix_symbol=sym,
                savepath=(os.path.join(savedir, f"table_sel_{side}_{method}.tex")
                          if savedir else None))
            if savedir:   # full-length CSV (all features, not only top_n)
                full = feature_table_named(res, side, names, n_orig,
                                           top_n=10**9, prevalence=prev,
                                           labels=labels, sort_norm=sort_norm,
                                           relative=relative,
                                           tie_break=(tie_break if prev is not
                                                      None else "norm"))
                full.to_csv(os.path.join(savedir,
                            f"selection_{side}_{method}_full.csv"), index=False)
        cmp = feature_table_across_methods(collected, side, names, n_orig,
                                           top_n=top_n, labels=labels,
                                           rank_by=method, prevalence=prev,
                                           sort_norm=sort_norm, relative=relative)
        out[side + "_methods"] = cmp
        if len(cmp):
            out[side + "_methods_tex"] = feature_table_across_methods_latex(
                cmp, slabel, prec=prec,
                savepath=(os.path.join(savedir, f"table_sel_{side}_methods.tex")
                          if savedir else None))
        if verbose and len(tab):
            print(f"\n=== Top {len(tab)} {slabel} ({_label_for(method, collected)}) ===")
            show = tab.drop(columns=["feature_raw"])
            with pd.option_context("display.width", 140, "display.precision", 3):
                print(show.to_string(index=False))
    if verbose and savedir:
        print(f"\nTables/CSVs written to {savedir}")
    return out


# =============================================================================
# Drug-side (V) functional-group figures
# =============================================================================
# Driven by the selection tables (frequency / relative norm across fits), not
# by a single fit's coefficient matrix as in the earlier exploratory script.
#
#   tab = feature_table_named(res, "drug", names_V, N_ORIG_V, top_n=8, ...)
#   fig_fg_structures(tab["feature_raw"], annot=tab["freq_grid"], savepath=...)
#   fig_fg_importance(tab, savepath=...)
#   fig_fg_presence(prev_V, tab["feature_raw"], savepath=...)
#   make_drug_figures(tab, prev_V, FIG_DIR)       # all three at once

# representative structures for display (schematic, not the SMARTS patterns)
FG_DISPLAY = {
    "Aromatic_ring": "c1ccccc1",
    "Heteroaromatic": "c1ncccc1",
    "Aliphatic_ring": "C1CCCCC1",
    "Primary_amine": "CN",
    "Secondary_amine": "CNC",
    "Tertiary_amine": "CN(C)C",
    "Amide": "NC=O",
    "Nitro": "[N+](=O)[O-]",
    "Nitrile": "C#N",
    "Carboxylic_acid": "C(=O)O",
    "Ester": "CC(=O)OC",
    "Alcohol": "CO",
    "Phenol": "c1ccccc1O",
    "Thiol": "CS",
    "Thioether": "CSC",
    "Sulfoxide_sulfone": "CS(=O)C",
    "Any_halogen": "CF",
    "Fluorine": "CF",
    "Chlorine": "CCl",
    "Bromine": "CBr",
    "Iodine": "CI",
    "Aromatic_halogen": "c1ccccc1F",
    "Ether": "COC",
    "Imine": "C=N",
    "Lactam": "O=C1NCCC1",
    "Lactone": "O=C1OCCC1",
    "Carboxamide_NH": "C(=O)N",
    "Sulfonamide": "CS(=O)(=O)NC",
    "Basic_center": "CN=C",
    "Acidic_center": "C(=O)O",
    "High_polar_N_O_count": "NCCO",
    "Aromatic_nitro": "c1ccccc1[N+](=O)[O-]",
    "Macrocycle": "C1CCCCCCCC1",
}

# display labels for conceptual (non-literal) groups
FG_DISPLAY_LABELS = {
    "Basic_center": "Basic amine center",
    "Acidic_center": "Acidic center",
    "High_polar_N_O_count": "Polar N/O count (>=3)",
    "Primary_amine": "Primary amine (NH2)",
    "Secondary_amine": "Secondary amine (NH)",
    "Tertiary_amine": "Tertiary amine",
    "Carboxamide_NH": "Carboxamide (NH)",
    "Sulfoxide_sulfone": "Sulfoxide / sulfone",
}


_RDKIT_SAFE = str.maketrans({
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
    "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
    "\u2265": ">=", "\u2264": "<=", "\u2013": "-", "\u2014": "-", "\u2026": "...",
    "\u2212": "-", "\u00d7": "x", "\u2019": "'", "\u201c": '"', "\u201d": '"',
})


def _rdkit_safe(text):
    """RDKit's drawing font lacks many Unicode glyphs (subscripts, >=, ...),
    which render as blank gaps; map them to ASCII equivalents."""
    t = str(text).translate(_RDKIT_SAFE)
    return t.encode("ascii", "ignore").decode()   # drop anything else exotic


_SUB_RE = None


def _subscript_formula(text):
    """'NH2' -> 'NH$_2$', 'C(=O)NH2' -> 'C(=O)NH$_2$' for matplotlib mathtext.
    A digit run directly following a letter or ')' is treated as a chemical
    subscript; digits after spaces, '(', '=', '>' etc. are left alone, so
    annotations like '(1.00)' and '>=3' are unaffected."""
    import re
    global _SUB_RE
    if _SUB_RE is None:
        _SUB_RE = re.compile(r"(?<=[A-Za-z\)])(\d+)")
    return _SUB_RE.sub(lambda m: f"$_{{{m.group(1)}}}$", str(text))


def fig_fg_structures(features, annot=None, annot_fmt="{:.2f}", labels=None,
                      fg_display=None, mols_per_row=4, sub_img_size=(220, 220),
                      savepath=None, use_svg=True, backend="mpl",
                      subscript_digits=True, legend_fontsize=10,
                      annot_label=None):
    """Grid of representative structures for the selected functional groups.

    features : iterable of raw feature names (keys of fg_display)
    annot    : optional iterable aligned with features, appended to the legend
               in parentheses (e.g. tab['freq_grid'] = selection frequency)
    labels   : display-label overrides on top of FG_DISPLAY_LABELS
    backend  : 'mpl'   (default) RDKit draws the molecules, matplotlib draws
                       the legends -> real subscripts (NH$_2$), journal fonts,
                       vector text; savepath may be .pdf/.png/.svg
               'rdkit' pure RDKit MolsToGridImage (plain-text legends, ASCII
                       sanitized); savepath .svg (use_svg) or .png
    subscript_digits : with backend='mpl', render formula digits as subscripts
    annot_label      : optional short note under the grid, e.g.
                       "values: selection frequency over the HP grid"
    Returns the matplotlib Figure ('mpl') or the RDKit image ('rdkit').
    Groups without a display structure are skipped with a notice.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError as e:
        raise ImportError("fig_fg_structures needs rdkit "
                          "(conda install -c conda-forge rdkit)") from e
    fg_display = fg_display or FG_DISPLAY
    labels = {**FG_DISPLAY_LABELS, **(labels or {})}
    features = list(features)
    annot = list(annot) if annot is not None else None
    if annot is not None and len(annot) != len(features):
        raise ValueError(f"annot has {len(annot)} entries but features has "
                         f"{len(features)}; they must be aligned.")

    mols, legends, skipped = [], [], []
    for i, f in enumerate(features):
        smi = fg_display.get(f)
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            skipped.append(f)
            continue
        leg = pretty_name(f, labels)
        if annot is not None:
            leg += f" ({annot_fmt.format(annot[i])})"
        mols.append(mol)
        legends.append(leg)
    if skipped:
        print("fig_fg_structures: no display structure for", skipped)
    if not mols:
        raise ValueError("No drawable functional groups.")

    if backend == "rdkit":
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size,
                                   legends=[_rdkit_safe(l) for l in legends],
                                   useSVG=use_svg)
        if savepath:
            _save_rdkit_image(img, savepath, use_svg)
        return img

    # ---- matplotlib backend: molecule rasters + mathtext legends ----
    n = len(mols)
    ncol = min(mols_per_row, n)
    nrow = int(np.ceil(n / ncol))
    px = max(sub_img_size)
    # two-line legend: name on top, annotation below -> no horizontal crowding
    names = [pretty_name(f, labels) for f in features if f not in skipped]
    annots = ([annot_fmt.format(a) for a, f in zip(annot, features)
               if f not in skipped] if annot is not None else None)
    longest = max(len(nm) for nm in names)
    cell_in = max(1.9, 0.085 * longest * legend_fontsize / 10 + 0.4)
    cell_h = cell_in + 0.55
    note_h = 0.45 if annot_label else 0.0           # inches reserved for note
    fig_h = cell_h * nrow + note_h
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(cell_in * ncol, fig_h),
                                 squeeze=False)
        for k, ax in enumerate(axes.ravel()):
            ax.axis("off")
            if k >= n:
                continue
            im = Draw.MolToImage(mols[k], size=(px * 2, px * 2))   # 2x for crispness
            ax.imshow(np.asarray(im), interpolation="lanczos")
            name = _subscript_formula(names[k]) if subscript_digits else names[k]
            ax.text(0.5, -0.04, name, transform=ax.transAxes, ha="center",
                    va="top", fontsize=legend_fontsize, color=_INK)
            if annots is not None:
                ax.text(0.5, -0.20, f"({annots[k]})", transform=ax.transAxes,
                        ha="center", va="top", fontsize=legend_fontsize - 1,
                        color=_MUTED)
        # bottom margin = space for the last row's 2-line legend + the note
        legend_h = 0.55
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99,
                            bottom=(legend_h + note_h) / fig_h,
                            wspace=0.08, hspace=0.45)
        if annot_label:
            fig.text(0.5, 0.3 * note_h / fig_h, annot_label, ha="center",
                     va="center", fontsize=8.5, color=_MUTED)
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


def _save_rdkit_image(img, savepath, use_svg):
    """RDKit's MolsToGridImage returns different objects depending on the
    environment: str (SVG), PIL.Image (PNG, scripts), or IPython.display.Image
    (PNG/SVG, notebooks; has .data, no .save). Handle all of them."""
    data = getattr(img, "data", None)
    if isinstance(img, str):                      # raw SVG text
        data = img
    if data is not None:
        mode = "w" if isinstance(data, str) else "wb"
        with open(savepath, mode) as fh:
            fh.write(data)
    elif hasattr(img, "save"):                    # PIL image
        img.save(savepath)
    else:
        raise TypeError(f"Don't know how to save RDKit image of type {type(img)}")


def fig_fg_importance(tab, metric="freq_grid", secondary="freq_selhp",
                      show_norm=True, title=None, figsize=None, savepath=None):
    """Horizontal bars of selection frequency for the features in `tab`
    (output of feature_table_named, drug side). Replaces the old single-fit
    'max |coefficient|' bar chart with a stability-based importance.

    metric    : bar length: 'freq_grid' (pooled over HP grid) or 'freq_selhp'
    secondary : optional second frequency drawn as a marker on each bar
    show_norm : relative norm (median) listed in a column at the right
    Rows keep the table order (rank) within ties.
    """
    t = tab.copy()
    t["_r"] = -np.arange(len(t))                  # preserve table order in ties
    t = t.sort_values([metric, "_r"], ascending=[True, True])
    n = len(t)
    figsize = figsize or (6.4, 0.42 * n + 1.5)
    lab = {"freq_grid": "Freq. (HP grid)", "freq_selhp": "Freq. (sel. HP)"}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(n)
        ax.barh(y, t[metric].values, color="#C0392B", alpha=0.85,
                edgecolor="none", label=lab.get(metric, metric))
        if secondary and secondary in t.columns:
            ax.plot(t[secondary].values, y, linestyle="none", marker="o",
                    markersize=6, markerfacecolor="white", markeredgewidth=1.6,
                    color=_INK, label=lab.get(secondary, secondary))
        xmax = 1.0
        if show_norm and "norm" in t.columns:
            xmax = 1.22
            for yi, nm in enumerate(t["norm"].values):
                ax.text(1.07, yi, f"{nm:.2f}", va="center", ha="left",
                        fontsize=9, color=_MUTED)
            ax.text(1.07, n - 0.45, "rel. norm", fontsize=8.5, color=_MUTED,
                    ha="left", va="bottom")
            ax.axvline(1.0, color=_GRID, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(t["feature"].values, fontsize=10)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, n - 0.4 + (0.5 if show_norm else 0))
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("Selection frequency", fontsize=11)
        ax.set_title(title or "Selected drug functional groups", fontsize=12,
                     pad=8)
        _style_axis(ax)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend(fontsize=9, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.45, -0.28), ncol=2)
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


def fig_fg_presence(prevalence, features, labels=None, entity_label="Drug",
                    title=None, figsize=None, savepath=None, cmap=None):
    """Binary presence heatmap: entities (drugs) x selected functional groups,
    from the entities x features DataFrame returned by load_feature_names().
    Rows are ordered by how many of the selected groups each drug carries.
    """
    labels = {**FG_DISPLAY_LABELS, **(labels or {})}
    feats = [f for f in features if f in prevalence.columns]
    sub = prevalence[feats].astype(float)
    sub = sub.loc[sub.sum(axis=1).sort_values(ascending=False).index]
    n_e, n_f = sub.shape
    figsize = figsize or (0.55 * n_f + 2.2, 0.32 * n_e + 1.5)
    if cmap is None:
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#f3f4f6", "#C0392B"])     # absent / present
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(sub.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(n_f))
        ax.set_xticklabels([pretty_name(f, labels) for f in feats],
                           rotation=90, fontsize=9)
        ax.set_yticks(range(n_e))
        ax.set_yticklabels(sub.index, fontsize=8)
        ax.set_xticks(np.arange(-.5, n_f, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_e, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title or f"Selected functional groups across {entity_label.lower()}s",
                     fontsize=12, pad=8)
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


def make_drug_figures(tab, prevalence=None, figdir=".", top_n=8,
                      metric="freq_grid", labels=None, prefix="fig_fg",
                      formats=("pdf", "png"), verbose=True):
    """Write the three drug-side figures for the top_n rows of `tab`:
    structures grid (legend = name (selection frequency)), importance bars,
    presence heatmap. `metric` is the frequency shown in the structure legends
    and used for the bars: 'freq_grid' (pooled over the HP grid) or
    'freq_selhp' (outer folds at the CV-selected HP)."""
    os.makedirs(figdir, exist_ok=True)
    t = tab.head(top_n)
    out = {}
    note = {"freq_grid": "values: selection frequency over trials x folds x HP grid",
            "freq_selhp": "values: selection frequency over outer folds at the CV-selected HP",
            "prevalence_pct": "values: prevalence percentage"
            }.get(metric, f"values: {metric}")
    try:
        fig = fig_fg_structures(t["feature_raw"], annot=t[metric], labels=labels,
                                backend="mpl", annot_label=note)
        for ext in formats:
            fig.savefig(os.path.join(figdir, f"{prefix}_structures.{ext}"),
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
        out["structures"] = True
    except ImportError as e:
        print(e)
    fig = fig_fg_importance(t, metric=metric)
    for ext in formats:
        fig.savefig(os.path.join(figdir, f"{prefix}_importance.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    out["importance"] = True
    if prevalence is not None:
        fig = fig_fg_presence(prevalence, t["feature_raw"], labels=labels)
        for ext in formats:
            fig.savefig(os.path.join(figdir, f"{prefix}_presence.{ext}"),
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
        out["presence"] = True
    if verbose:
        print(f"Drug-side figures written to {figdir}: "
              f"{sorted(f for f in os.listdir(figdir) if f.startswith(prefix))}")
    return out