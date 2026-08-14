

"""
Spatial proximity effect analysis (Visium HD real cell types)
- Computation and plotting are separated, supporting repeated plotting without recalculation
- Asterisks fit tightly to bar ends, p>=0.05 displayed as 'ns'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import ttest_ind
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from anndata import AnnData
except ImportError:
    AnnData = type('AnnData', (), {})

# ---------- Helper functions ----------
def extract_obs_from_adata(adata, use_raw=False):
    if use_raw and hasattr(adata, 'raw') and hasattr(adata.raw, 'obs'):
        return adata.raw.obs.copy()
    elif hasattr(adata, 'obs'):
        return adata.obs.copy()
    else:
        raise ValueError("Input object does not have obs attribute")

def get_drug_column_name(adata, drug_name, sample_id="CRC1"):
    full_name = f"{sample_id}_{drug_name}"
    if hasattr(adata, 'obs') and full_name in adata.obs.columns:
        return full_name
    else:
        if hasattr(adata, 'obs'):
            for col in adata.obs.columns:
                if drug_name.lower() in col.lower():
                    return col
        return full_name

# ---------- Compute distance to effector cell type ----------
def add_distance_to_celltype_visium(df, effector_ct, coord_cols=['array_col', 'array_row'],
                                    celltype_col='DeconvolutionLabel1'):
    """
    Compute Euclidean distance from each spot to the nearest spot of a given effector cell type
    """
    df = df.copy()
    effector_mask = df[celltype_col] == effector_ct
    if effector_mask.sum() == 0:
        raise ValueError(f"No spots with cell type '{effector_ct}'")
    effector_coords = df.loc[effector_mask, coord_cols].values
    all_coords = df[coord_cols].values
    tree = KDTree(effector_coords)
    distances, _ = tree.query(all_coords, k=1)
    df[f'dist_to_{effector_ct}'] = distances
    return df

# ---------- Permutation test ----------
def permutation_test_for_pair_visium(df, effector_ct, target_ct,
                                     target_col, cutoff, comparison, min_pairs,
                                     coord_cols, celltype_col, n_perm=50):
    """
    Permutation test for a single effector-target pair
    """
    perm_effects = []
    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm[celltype_col] = np.random.permutation(df_perm[celltype_col].values)
        try:
            df_perm = add_distance_to_celltype_visium(df_perm, effector_ct, coord_cols, celltype_col)
        except ValueError:
            continue
        res = compute_pair_effect_visium(
            df_perm, effector_ct, target_ct,
            target_col=target_col,
            cutoff=cutoff,
            comparison=comparison,
            min_pairs=min_pairs,
            coord_cols=coord_cols,
            celltype_col=celltype_col,
            verbose=False
        )
        if res is not None and not np.isnan(res['cohens_d']):
            perm_effects.append(res['cohens_d'])
    return perm_effects

# ---------- Compute proximity effect for a single pair ----------
def compute_pair_effect_visium(df, effector_ct, target_ct,
                               target_col='CRC2_Irinotecan',
                               cutoff=None,
                               comparison='farthest',
                               min_pairs=20,
                               coord_cols=['array_col', 'array_row'],
                               celltype_col='DeconvolutionLabel1',
                               verbose=False):
    """
    Compute proximity effect (Cohen's d) between one effector cell type and one target cell type
    """
    df_target = df[df[celltype_col] == target_ct].copy()
    if len(df_target) < min_pairs:
        return None

    dist_col = f'dist_to_{effector_ct}'
    if dist_col not in df_target.columns:
        df_all = df.copy()
        if dist_col not in df_all.columns:
            df_all = add_distance_to_celltype_visium(df_all, effector_ct, coord_cols, celltype_col)
        df_target = df_all[df_all[celltype_col] == target_ct].copy()

    if cutoff is None:
        cutoff = np.nanmedian(df_target[dist_col].values)
        if cutoff == 0:
            cutoff = np.nanpercentile(df_target[dist_col], 75)
            if cutoff == 0:
                cutoff = np.nanmax(df_target[dist_col]) * 0.5

    near_mask = df_target[dist_col] < cutoff
    near_indices = np.where(near_mask)[0]
    far_indices = np.where(~near_mask)[0]
    n_near = len(near_indices)
    if n_near < min_pairs or len(far_indices) < n_near:
        return None

    if comparison == 'farthest':
        far_dists = df_target[dist_col].values[far_indices]
        sorted_idx = np.argsort(far_dists)[::-1]
        far_selected = far_indices[sorted_idx[:n_near]]
    else:
        np.random.seed(42)
        far_selected = np.random.choice(far_indices, n_near, replace=False)

    near_vals = df_target[target_col].values[near_indices]
    far_vals = df_target[target_col].values[far_selected]
    near_clean = near_vals[~np.isnan(near_vals)]
    far_clean = far_vals[~np.isnan(far_vals)]
    if len(near_clean) < min_pairs or len(far_clean) < min_pairs:
        return None

    t_stat, p_val = ttest_ind(near_clean, far_clean, equal_var=False)
    n1, n2 = len(near_clean), len(far_clean)
    mean1, mean2 = np.mean(near_clean), np.mean(far_clean)
    std1, std2 = np.std(near_clean, ddof=1), np.std(far_clean, ddof=1)
    pooled_sd = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    cohens_d = (mean1 - mean2) / pooled_sd if pooled_sd > 0 else np.nan

    return {
        'effector': effector_ct,
        'target': target_ct,
        'cohens_d': cohens_d,
        'p_value': p_val,
        'n_near': n1,
        'n_far': n2,
        'cutoff': cutoff
    }

# ---------- Main computation function ----------
def compute_proximity_visium(adata,
                             drug_name="Irinotecan",
                             sample_id="CRC1",
                             effector_celltypes=None,
                             target_celltypes=None,
                             celltype_col='DeconvolutionLabel1',
                             cutoff=None,
                             comparison='farthest',
                             min_pairs=20,
                             n_perm=50,
                             use_raw_obs=False,
                             verbose=True,
                             save_path=None):
    """
    Compute proximity effects for all effector-target cell type pairs
    """
    if verbose:
        print("="*60)
        print(f"Computing proximity effects for {sample_id} - {drug_name}")
        print("="*60)

    df = extract_obs_from_adata(adata, use_raw=use_raw_obs)
    if verbose:
        print(f"  Data shape: {df.shape}")

    if celltype_col not in df.columns:
        raise ValueError(f"Column '{celltype_col}' not found in adata.obs.")

    drug_col = get_drug_column_name(adata, drug_name, sample_id)
    if drug_col not in df.columns:
        raise ValueError(f"Drug column {drug_col} not found")
    if verbose:
        print(f"  Target column: {drug_col}")

    all_types = df[celltype_col].unique().tolist()
    if verbose:
        print(f"  Found {len(all_types)} cell types.")

    if effector_celltypes is None:
        effector_celltypes = all_types
    else:
        effector_celltypes = [ct for ct in effector_celltypes if ct in all_types]
        if len(effector_celltypes) == 0:
            raise ValueError("No valid effector cell types provided.")
    if target_celltypes is None:
        target_celltypes = all_types
    else:
        target_celltypes = [ct for ct in target_celltypes if ct in all_types]
        if len(target_celltypes) == 0:
            raise ValueError("No valid target cell types provided.")

    if verbose:
        print(f"  Effector types: {effector_celltypes}")
        print(f"  Target types: {target_celltypes}")

    all_results = []
    perm_dict = {}

    df_temp = df.copy()
    for effector_ct in effector_celltypes:
        dist_col = f'dist_to_{effector_ct}'
        if dist_col not in df_temp.columns:
            df_temp = add_distance_to_celltype_visium(df_temp, effector_ct,
                                                      coord_cols=['array_col', 'array_row'],
                                                      celltype_col=celltype_col)

    for effector_ct in effector_celltypes:
        if verbose:
            print(f"\nEffector: {effector_ct}")
        dist_col = f'dist_to_{effector_ct}'
        for target_ct in target_celltypes:
            if verbose:
                print(f"  → Target: {target_ct}", end=' ')
            df_target = df_temp[df_temp[celltype_col] == target_ct].copy()
            if len(df_target) < min_pairs:
                if verbose:
                    print(f"(too few cells: {len(df_target)})")
                continue

            if cutoff is None:
                cutoff_local = np.nanmedian(df_target[dist_col].values)
                if cutoff_local == 0:
                    cutoff_local = np.nanpercentile(df_target[dist_col], 75)
                    if cutoff_local == 0:
                        cutoff_local = np.nanmax(df_target[dist_col]) * 0.5
            else:
                cutoff_local = cutoff

            near_mask = df_target[dist_col] < cutoff_local
            near_indices = np.where(near_mask)[0]
            far_indices = np.where(~near_mask)[0]
            n_near = len(near_indices)
            if n_near < min_pairs or len(far_indices) < n_near:
                if verbose:
                    print("(insufficient pairs)")
                continue

            if comparison == 'farthest':
                far_dists = df_target[dist_col].values[far_indices]
                sorted_idx = np.argsort(far_dists)[::-1]
                far_selected = far_indices[sorted_idx[:n_near]]
            else:
                np.random.seed(42)
                far_selected = np.random.choice(far_indices, n_near, replace=False)

            near_vals = df_target[drug_col].values[near_indices]
            far_vals = df_target[drug_col].values[far_selected]
            near_clean = near_vals[~np.isnan(near_vals)]
            far_clean = far_vals[~np.isnan(far_vals)]
            if len(near_clean) < min_pairs or len(far_clean) < min_pairs:
                if verbose:
                    print("(insufficient after NaN removal)")
                continue

            t_stat, p_val = ttest_ind(near_clean, far_clean, equal_var=False)
            n1, n2 = len(near_clean), len(far_clean)
            mean1, mean2 = np.mean(near_clean), np.mean(far_clean)
            std1, std2 = np.std(near_clean, ddof=1), np.std(far_clean, ddof=1)
            pooled_sd = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
            cohens_d = (mean1 - mean2) / pooled_sd if pooled_sd > 0 else np.nan

            res = {
                'effector': effector_ct,
                'target': target_ct,
                'cohens_d': cohens_d,
                'p_value': p_val,
                'n_near': n1,
                'n_far': n2,
                'cutoff': cutoff_local
            }
            all_results.append(res)
            pair_key = f"{effector_ct}→{target_ct}"

            if n_perm > 0:
                perm_effects = permutation_test_for_pair_visium(
                    df, effector_ct, target_ct,
                    target_col=drug_col,
                    cutoff=cutoff_local,
                    comparison=comparison,
                    min_pairs=min_pairs,
                    coord_cols=['array_col', 'array_row'],
                    celltype_col=celltype_col,
                    n_perm=n_perm
                )
                perm_dict[pair_key] = perm_effects
            else:
                perm_dict[pair_key] = [0]

            if verbose:
                print(f"n_near={n1}, n_far={n2}, d={cohens_d:.3f}")

    if len(all_results) == 0:
        print("No valid results for any effector-target pair.")
        return None, None

    prox_df = pd.DataFrame(all_results)
    prox_df['pair'] = prox_df['effector'] + ' → ' + prox_df['target']

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump({'prox_df': prox_df, 'perm_dict': perm_dict}, f)
        if verbose:
            print(f"Results saved to {save_path}")

    return prox_df, perm_dict

# ---------- Plotting function (supports 'ns' for p>=0.05) ----------
def plot_proximity_visium(prox_df, perm_dict,
                          sample_id="CRC1",
                          drug_name="Irinotecan",
                          save_figures=True,
                          output_dir="figure_outputs",
                          figsize=None,
                          left_margin=0.2,
                          bar_height=0.6,
                          title_fontsize=12,
                          label_fontsize=10,
                          star_fontsize=10,
                          ns_fontsize=8,
                          star_offset_scale=0.02,
                          x_axis_extra=0.02,
                          xlim_left=None,
                          xlim_right=None,
                          save_filename=None):
    """
    Plot dual-panel summary figure with observed effects and permutation confidence intervals
    """
    if len(prox_df) == 0:
        print("No data to plot.")
        return

    plot_df = prox_df.sort_values('cohens_d', ascending=False)
    ordered_pairs = plot_df['pair'].tolist()
    scores = plot_df['cohens_d'].values
    pvals = plot_df['p_value'].values

    perm_summary = pd.DataFrame({
        "median": [np.median(perm_dict.get(pair, [0])) for pair in ordered_pairs],
        "low": [np.percentile(perm_dict.get(pair, [0]), 2.5) if len(perm_dict.get(pair, [])) > 1 else -0.1 for pair in ordered_pairs],
        "high": [np.percentile(perm_dict.get(pair, [0]), 97.5) if len(perm_dict.get(pair, [])) > 1 else 0.1 for pair in ordered_pairs]
    }, index=ordered_pairs)

    if figsize is None:
        figsize = (14, 0.5 * len(ordered_pairs) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    y_pos = np.arange(len(ordered_pairs))

    # ---- Left panel: Observed effect sizes ----
    colors = ['#d95f5f' if v > 0 else '#5f8fd9' for v in scores]
    axes[0].barh(y_pos, scores, color=colors, height=bar_height)
    axes[0].axvline(0, color='gray', linestyle='--')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ordered_pairs)
    axes[0].tick_params(axis='y', pad=2)
    axes[0].set_xlabel("Cell proximity score (Cohen's d)", fontsize=label_fontsize)
    axes[0].set_title(f"Observed cell proximity effect of {drug_name}", fontsize=title_fontsize)

    # ---- Add significance labels (asterisks or 'ns') ----
    star_x_positions = []
    for i, (score, p) in enumerate(zip(scores, pvals)):
        if p < 0.001:
            label = '***'
        elif p < 0.01:
            label = '**'
        elif p < 0.05:
            label = '*'
        else:
            label = 'ns'
            offset = star_offset_scale * max(1, abs(score))
            x = score + (offset if score > 0 else -offset)
            ha = 'left' if score > 0 else 'right'
            axes[0].text(x, i, label, va='center', ha=ha, fontsize=ns_fontsize, color='black')
            star_x_positions.append(x)
            continue

        offset = star_offset_scale * max(1, abs(score))
        x = score + (offset if score > 0 else -offset)
        ha = 'left' if score > 0 else 'right'
        axes[0].text(x, i, label, va='center', ha=ha, fontsize=star_fontsize, color='black')
        star_x_positions.append(x)

    # Auto-expand x-axis if not manually set
    if xlim_left is None and star_x_positions:
        xmin, xmax = axes[0].get_xlim()
        max_star = max(star_x_positions)
        min_star = min(star_x_positions)
        if max_star > xmax:
            axes[0].set_xlim(xmin, max_star + x_axis_extra * abs(max_star) + 0.05)
        if min_star < xmin:
            axes[0].set_xlim(min_star - x_axis_extra * abs(min_star) - 0.05, xmax)
    elif xlim_left is not None:
        axes[0].set_xlim(xlim_left)

    # ---- Right panel: Permutation confidence intervals ----
    axes[1].errorbar(
        perm_summary["median"], y_pos,
        xerr=[perm_summary["median"] - perm_summary["low"],
              perm_summary["high"] - perm_summary["median"]],
        fmt='o', color='black', ecolor='gray', capsize=3
    )
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ordered_pairs)
    axes[1].tick_params(axis='y', pad=2)
    axes[1].set_title("Spatially permuted", fontsize=title_fontsize)
    axes[1].set_xlabel("Median proximity score (95% CI)", fontsize=label_fontsize)
    if xlim_right is not None:
        axes[1].set_xlim(xlim_right)
    else:
        axes[1].set_xlim(-1, 1)

    plt.subplots_adjust(left=left_margin)
    plt.tight_layout()

    if save_figures:
        os.makedirs(output_dir, exist_ok=True)
        if save_filename is None:
            save_filename = f"{sample_id}_{drug_name}_summary_dual.pdf"
        path = os.path.join(output_dir, save_filename)
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Figure saved: {path}")
    plt.show()

# ---------- Backward compatible interface ----------
def analyze_proximity_visium(adata,
                             drug_name="Irinotecan",
                             sample_id="CRC1",
                             effector_celltypes=None,
                             target_celltypes=None,
                             celltype_col='DeconvolutionLabel1',
                             cutoff=None,
                             comparison='farthest',
                             min_pairs=20,
                             n_perm=50,
                             use_raw_obs=False,
                             save_figures=True,
                             output_dir="figure_outputs",
                             verbose=True,
                             figsize=None,
                             left_margin=0.2,
                             save_path=None):
    """
    One-step function: compute + plot proximity effects
    """
    prox_df, perm_dict = compute_proximity_visium(
        adata=adata,
        drug_name=drug_name,
        sample_id=sample_id,
        effector_celltypes=effector_celltypes,
        target_celltypes=target_celltypes,
        celltype_col=celltype_col,
        cutoff=cutoff,
        comparison=comparison,
        min_pairs=min_pairs,
        n_perm=n_perm,
        use_raw_obs=use_raw_obs,
        verbose=verbose,
        save_path=save_path
    )
    if prox_df is not None:
        plot_proximity_visium(
            prox_df, perm_dict,
            sample_id=sample_id,
            drug_name=drug_name,
            save_figures=save_figures,
            output_dir=output_dir,
            figsize=figsize,
            left_margin=left_margin
        )
        return {'prox_df': prox_df, 'perm_scores': perm_dict}
    else:
        return None