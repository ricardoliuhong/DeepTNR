import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import ttest_ind
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from anndata import AnnData
except ImportError:
    AnnData = type('AnnData', (), {})

# ---------- Core function: calculate distance to effector cells ----------
def add_distance_to_celltype(df, effector_cell_type, cell_type_prefix='q95_spot_factors',
                             coord_cols=['array_col', 'array_row'], threshold=0):
    df = df.copy()
    effector_col = f"{cell_type_prefix}{effector_cell_type}"
    if effector_col not in df.columns:
        raise ValueError(f"Column {effector_col} not found.")
    effector_mask = df[effector_col] > threshold
    if effector_mask.sum() == 0:
        raise ValueError(f"No spots with {effector_cell_type} abundance > {threshold}")
    effector_coords = df.loc[effector_mask, coord_cols].values
    all_coords = df[coord_cols].values
    tree = KDTree(effector_coords)
    distances, _ = tree.query(all_coords, k=1)
    df[f'dist_to_{effector_cell_type}'] = distances
    return df

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

def extract_cell_types(df, prefix='q95_spot_factors'):
    cell_type_cols = [c for c in df.columns if c.startswith(prefix) and df[c].sum() > 0]
    cell_types = [c.replace(prefix, "") for c in cell_type_cols]
    return cell_types, cell_type_cols

# ---------- Permutation test ----------
def permutation_test_for_pair(df, effector_ct, target_ct,
                              target_col, effector_quantile, target_quantile,
                              cutoff, comparison, min_pairs,
                              cell_type_prefix, coord_cols, n_perm=50):
    perm_effects = []
    effector_col_name = f"{cell_type_prefix}{effector_ct}"
    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm[effector_col_name] = np.random.permutation(df_perm[effector_col_name].values)
        res = compute_pair_effect_custom(
            df_perm, effector_ct, target_ct,
            target_col=target_col,
            effector_quantile=effector_quantile,
            target_quantile=target_quantile,
            cutoff=cutoff,
            comparison=comparison,
            min_pairs=min_pairs,
            cell_type_prefix=cell_type_prefix,
            coord_cols=coord_cols,
            verbose=False
        )
        if res is not None and not np.isnan(res['cohens_d']):
            perm_effects.append(res['cohens_d'])
    return perm_effects

# ---------- Calculate proximity effect for a single pair ----------
def compute_pair_effect_custom(df, effector_ct, target_ct,
                               target_col='CRC2_Irinotecan',
                               effector_quantile=0.75,
                               target_quantile=0.5,
                               cutoff=None,
                               comparison='farthest',
                               min_pairs=20,
                               cell_type_prefix='q95_spot_factors',
                               coord_cols=['array_col', 'array_row'],
                               verbose=False):
    # 1. Filter target cells
    df_work = df.copy()
    target_col_name = f"{cell_type_prefix}{target_ct}"
    if target_col_name not in df_work.columns:
        raise ValueError(f"Column {target_col_name} not found.")
    if target_quantile is None:
        raise ValueError("target_quantile must be a numeric value.")
    thresh = df_work[target_col_name].quantile(target_quantile)
    df_work = df_work[df_work[target_col_name] > thresh].copy()
    if len(df_work) < min_pairs:
        return None

    # 2. Calculate distance to effector cells
    effector_col = f"{cell_type_prefix}{effector_ct}"
    if effector_col not in df_work.columns:
        raise ValueError(f"Column {effector_col} not found.")
    threshold = df_work[effector_col].quantile(effector_quantile)
    df_work = add_distance_to_celltype(df_work, effector_ct,
                                       cell_type_prefix=cell_type_prefix,
                                       coord_cols=coord_cols,
                                       threshold=threshold)

    dist_col = f'dist_to_{effector_ct}'

    # 3. Determine cutoff
    if cutoff is None:
        cutoff = np.nanmedian(df_work[dist_col].values)
        if cutoff == 0:
            cutoff = np.nanpercentile(df_work[dist_col], 75)
            if cutoff == 0:
                cutoff = np.nanmax(df_work[dist_col]) * 0.5

    # 4. Group near/far neighbors
    near_mask = df_work[dist_col] < cutoff
    near_indices = np.where(near_mask)[0]
    far_indices = np.where(~near_mask)[0]
    n_near = len(near_indices)
    if n_near < min_pairs or len(far_indices) < n_near:
        return None

    # Match farthest neighbors
    if comparison == 'farthest':
        far_dists = df_work[dist_col].values[far_indices]
        sorted_idx = np.argsort(far_dists)[::-1]
        far_selected = far_indices[sorted_idx[:n_near]]
    else:
        np.random.seed(42)
        far_selected = np.random.choice(far_indices, n_near, replace=False)

    near_vals = df_work[target_col].values[near_indices]
    far_vals = df_work[target_col].values[far_selected]
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

# ---------- Main analysis function ----------
def analyze_spatial_proximity_effect(adata,
                                     drug_name="Irinotecan",
                                     sample_id="CRC1",
                                     effector_celltypes=None,
                                     target_celltypes=None,
                                     effector_quantile=0.75,
                                     target_quantile=0.5,
                                     cutoff=None,
                                     comparison='farthest',
                                     min_pairs=20,
                                     n_perm=50,
                                     use_raw_obs=False,
                                     save_figures=True,
                                     output_dir="figure_outputs",
                                     verbose=True):
    if verbose:
        print("="*60)
        print(f"Custom proximity analysis (summary) for {sample_id} - {drug_name}")
        print("="*60)

    df = extract_obs_from_adata(adata, use_raw=use_raw_obs)
    if verbose:
        print(f"  Data shape: {df.shape}")

    drug_col = get_drug_column_name(adata, drug_name, sample_id)
    if drug_col not in df.columns:
        raise ValueError(f"Drug column {drug_col} not found")
    if verbose:
        print(f"  Target column: {drug_col}")

    all_types, _ = extract_cell_types(df, prefix='q95_spot_factors')
    if verbose:
        print(f"  Found {len(all_types)} cell types from q95 columns.")

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
    for effector_ct in effector_celltypes:
        if verbose:
            print(f"\nEffector: {effector_ct}")
        for target_ct in target_celltypes:
            if verbose:
                print(f"  -> Target: {target_ct}", end=' ')
            res = compute_pair_effect_custom(
                df, effector_ct, target_ct,
                target_col=drug_col,
                effector_quantile=effector_quantile,
                target_quantile=target_quantile,
                cutoff=cutoff,
                comparison=comparison,
                min_pairs=min_pairs,
                verbose=verbose
            )
            if res is not None:
                all_results.append(res)
                pair_key = f"{effector_ct}->{target_ct}"
                if n_perm > 0:
                    perm_effects = permutation_test_for_pair(
                        df, effector_ct, target_ct,
                        target_col=drug_col,
                        effector_quantile=effector_quantile,
                        target_quantile=target_quantile,
                        cutoff=res['cutoff'],
                        comparison=comparison,
                        min_pairs=min_pairs,
                        cell_type_prefix='q95_spot_factors',
                        coord_cols=['array_col', 'array_row'],
                        n_perm=n_perm
                    )
                    perm_dict[pair_key] = perm_effects
                else:
                    perm_dict[pair_key] = [0]
                if verbose:
                    print(f"n_near={res['n_near']}, n_far={res['n_far']}, d={res['cohens_d']:.3f}")
            else:
                if verbose:
                    print("(insufficient pairs)")

    if len(all_results) == 0:
        print("No valid results for any effector-target pair.")
        return None

    prox_df = pd.DataFrame(all_results)
    plot_dual_summary(prox_df, perm_dict, sample_id, drug_name, save_figures, output_dir)

    return {'prox_df': prox_df, 'perm_scores': perm_dict}

# ---------- Plotting function (only adds 'ns' annotation, otherwise unchanged) ----------
def plot_dual_summary(prox_df, perm_dict, sample_id, drug_name, save, output_dir):
    """
    Two-panel summary plot, strictly following the original plot_proximity_effect style.
    - Left panel: Effect size bar plot + significance stars (gray 'ns' for p>=0.05)
    - Right panel: Permutation confidence intervals (median +- 95% CI), x-axis fixed to [-1, 1]
    """
    if len(prox_df) == 0:
        print("No data to plot.")
        return

    # Create pair labels
    prox_df['pair'] = prox_df['effector'] + ' -> ' + prox_df['target']
    plot_df = prox_df.sort_values('cohens_d', ascending=False)
    ordered_pairs = plot_df['pair'].tolist()

    scores = plot_df['cohens_d'].values
    pvals = plot_df['p_value'].values

    # Build perm_summary (consistent with original format)
    perm_summary = pd.DataFrame({
        "median": [np.median(perm_dict.get(pair, [0])) for pair in ordered_pairs],
        "low": [np.percentile(perm_dict.get(pair, [0]), 2.5) if len(perm_dict.get(pair, [])) > 1 else -0.1 for pair in ordered_pairs],
        "high": [np.percentile(perm_dict.get(pair, [0]), 97.5) if len(perm_dict.get(pair, [])) > 1 else 0.1 for pair in ordered_pairs]
    }, index=ordered_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * len(ordered_pairs) + 1), sharey=True)
    y_pos = np.arange(len(ordered_pairs))

    # ---- Left panel: Effect size bar plot ----
    colors = ['#d95f5f' if v > 0 else '#5f8fd9' for v in scores]
    axes[0].barh(y_pos, scores, color=colors)
    axes[0].axvline(0, color='gray', linestyle='--')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ordered_pairs)
    axes[0].set_xlabel("Cell proximity score (Cohen's d)")
    axes[0].set_title(f"Observed cell proximity effect of {drug_name}")

    # Significance stars (show 'ns' for p>=0.05)
    for i, (score, p) in enumerate(zip(scores, pvals)):
        x_offset = 0.005 if score > 0 else -0.005
        ha = 'left' if score > 0 else 'right'
        
        if p < 0.001:
            label = '***'
            color = 'black'
        elif p < 0.01:
            label = '**'
            color = 'black'
        elif p < 0.05:
            label = '*'
            color = 'black'
        else:
            label = 'ns'
            color = 'black'
            x_offset = 0.005 if score > 0 else -0.005  # Keep consistent offset with stars
        
        axes[0].text(score + x_offset, i, label, va='center', ha=ha, fontsize=10, color=color)

    # ---- Right panel: Permutation confidence intervals (identical to original) ----
    axes[1].errorbar(
        perm_summary["median"], y_pos,
        xerr=[
            perm_summary["median"] - perm_summary["low"],
            perm_summary["high"] - perm_summary["median"]
        ],
        fmt='o', color='black', ecolor='gray', capsize=3
    )
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ordered_pairs)
    axes[1].set_title("Spatially permuted")
    axes[1].set_xlabel("Median proximity score (95% CI)")
    axes[1].set_xlim(-1, 1)   # Consistent with original

    plt.tight_layout()
    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{sample_id}_{drug_name}_summary_dual.pdf")
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Figure saved: {path}")
    plt.show()
