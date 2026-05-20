import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from scipy.spatial import KDTree
from scipy.stats import spearmanr
import os
import random
from typing import Optional, List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams

rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
sensitivity_colors = {
    'Sensitive': '#FF0000',
    'Resistant': '#0072B2',
    'Unknown': '#808080'
}

try:
    from anndata import AnnData
except ImportError:
    AnnData = Any
    print("Warning: anndata not installed, using Any type for AnnData")

# ============================================================
# SET RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================
def set_random_seeds(seed: int = 42):
    """
    Set random seeds for all random number generators to ensure reproducibility.
    
    Parameters:
    -----------
    seed : int
        Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Optional: if using tensorflow or pytorch, add those here
    # try:
    #     import torch
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # except ImportError:
    #     pass
    #     
    # try:
    #     import tensorflow as tf
    #     tf.random.set_seed(seed)
    # except ImportError:
    #     pass

# Set default seed at module level
DEFAULT_RANDOM_SEED = 42
set_random_seeds(DEFAULT_RANDOM_SEED)

def extract_obs_from_adata(adata, use_raw: bool = False) -> pd.DataFrame:
    if hasattr(adata, 'obs'):
        return adata.obs.copy()
    else:
        raise ValueError("Input object does not have obs attribute")

def get_drug_column_name(adata, drug_name: str, sample_id: str = "CRC1") -> str:
    full_name = f"{sample_id}_{drug_name}"
    if hasattr(adata, 'obs') and full_name in adata.obs.columns:
        return full_name
    else:
        if hasattr(adata, 'obs'):
            for col in adata.obs.columns:
                if drug_name.lower() in col.lower():
                    return col
        return full_name

def calculate_interface_distance(df: pd.DataFrame, 
                                 row_col: str = 'array_row', 
                                 col_col: str = 'array_col',
                                 region_col: str = 'Region_interface',
                                 interface_label: str = 'Interface',
                                 stroma_label: str = 'Stroma') -> pd.DataFrame:
    df = df.copy()
    df[row_col] = df[row_col].astype(int)
    df[col_col] = df[col_col].astype(int)
    rows = df[row_col].max() + 1
    cols = df[col_col].max() + 1
    grid = np.ones((rows, cols), dtype=bool)
    for r, c, region in zip(df[row_col], df[col_col], df[region_col]):
        if region == interface_label:
            grid[r, c] = False
    dist = distance_transform_edt(grid)
    spot_dist = []
    for r, c, region in zip(df[row_col], df[col_col], df[region_col]):
        d = dist[r, c]
        if region == stroma_label:
            d = -d
        spot_dist.append(d)
    df["interface_dist"] = spot_dist
    return df

def normalize_distance(df: pd.DataFrame, 
                       dist_col: str = 'interface_dist') -> pd.DataFrame:
    df = df.copy()
    max_abs = np.max(np.abs(df[dist_col]))
    if max_abs > 0:
        df[dist_col] = df[dist_col] / max_abs
    return df

def extract_cell_types(df: pd.DataFrame, 
                       prefix: str = 'q95_spot_factors') -> Tuple[List[str], List[str]]:
    cell_type_cols = [
        c for c in df.columns
        if c.startswith(prefix) and df[c].sum() > 0
    ]
    cell_types = [c.replace(prefix, "") for c in cell_type_cols]
    return cell_types, cell_type_cols

def calculate_spatial_density(df: pd.DataFrame,
                              cell_types: List[str],
                              cell_type_cols: List[str],
                              dist_col: str = 'interface_dist',
                              n_bins: int = 80,
                              smooth_sigma: float = 2.0,
                              normalize: bool = True) -> Tuple[pd.DataFrame, Dict]:
    bins = np.linspace(df[dist_col].min(), df[dist_col].max(), n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    density_data = {}
    tumor_side_mean = {}
    for ct, col in zip(cell_types, cell_type_cols):
        means = df.groupby(pd.cut(df[dist_col], bins, include_lowest=True))[col].mean().values
        means = np.nan_to_num(means)
        means = gaussian_filter1d(means, smooth_sigma)
        if normalize and means.max() > 0:
            means = means / means.max()
        density_data[ct] = means
        tumor_side_mean[ct] = np.mean(means[bin_centers > 0])
    density_df = pd.DataFrame(density_data)
    density_df["interface_dist"] = bin_centers
    return density_df, tumor_side_mean

def calculate_drug_gradient(df: pd.DataFrame,
                            drug_col: str,
                            dist_col: str = 'interface_dist',
                            n_bins: int = 80,
                            smooth_sigma: float = 2.0,
                            normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(df[dist_col].min(), df[dist_col].max(), n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    drug_mean = df.groupby(pd.cut(df[dist_col], bins, include_lowest=True))[drug_col].mean().values
    drug_mean = np.nan_to_num(drug_mean)
    drug_mean = gaussian_filter1d(drug_mean, smooth_sigma)
    if normalize and drug_mean.max() > drug_mean.min():
        drug_mean = (drug_mean - drug_mean.min()) / (drug_mean.max() - drug_mean.min())
    return bin_centers, drug_mean

def calculate_proximity_scores(df: pd.DataFrame,
                               cell_types: List[str],
                               cell_type_cols: List[str],
                               drug_col: str,
                               dist_col: str = 'interface_dist',
                               radius: float = 0.1,
                               min_neighbors: int = 3,
                               verbose: bool = False,
                               random_seed: int = 42) -> pd.DataFrame:
    """
    计算每种细胞类型的邻近效应评分
    公式: P_k = correlation(邻居中细胞类型k的丰度, 中心spot的药物敏感性)
    
    Parameters:
    -----------
    random_seed : int
        Random seed for reproducibility (default: 42)
    """
    # Set seed for this function
    np.random.seed(random_seed)
    
    df_copy = df.copy()
    
    if verbose:
        print(f"    Total spots: {len(df_copy)}")
        print(f"    Using radius: {radius}")
        print(f"    Min neighbors: {min_neighbors}")
        print(f"    Random seed: {random_seed}")
    
    # 获取坐标
    coords = df_copy[['array_col', 'array_row']].values
    
    # 归一化坐标，使半径更有意义
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords_range = coords_max - coords_min
    if np.max(coords_range) > 0:
        coords = (coords - coords_min) / np.max(coords_range)
    
    # 构建KDTree
    tree = KDTree(coords)
    
    if verbose:
        print("    Building KDTree...")
    
    # 预计算每个spot的邻居索引
    all_neighbors = []
    for i in range(len(df_copy)):
        indices = tree.query_ball_point(coords[i], radius)
        indices = [idx for idx in indices if idx != i]
        all_neighbors.append(indices)
    
    # 筛选有效spot
    valid_mask = [len(neighbors) >= min_neighbors for neighbors in all_neighbors]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    
    if verbose:
        print(f"    Spots with sufficient neighbors: {len(valid_indices)}/{len(df_copy)}")
    
    if len(valid_indices) < 10:
        if verbose:
            print("    Warning: Too few valid spots, returning zeros")
        prox_df = pd.DataFrame({
            'cell_type': cell_types,
            'proximity_score': [0] * len(cell_types)
        })
        prox_df = prox_df.sort_values('proximity_score')
        prox_df.set_index('cell_type', inplace=True)
        return prox_df
    
    # 获取药物敏感性
    y = df_copy[drug_col].values
    
    proximity_scores = []
    
    for ct, ct_col in zip(cell_types, cell_type_cols):
        if verbose:
            print(f"    Processing {ct}...")
        
        # 获取细胞类型k的丰度
        c = df_copy[ct_col].values
        
        # 计算每个有效spot的邻居平均丰度
        neighbor_abundance = np.zeros(len(df_copy))
        neighbor_abundance[:] = np.nan
        
        for i in valid_indices:
            neighbors = all_neighbors[i]
            if len(neighbors) > 0:
                neighbor_abundance[i] = np.mean(c[neighbors])
        
        # 只使用有效spot
        valid_y = y[valid_indices]
        valid_abundance = neighbor_abundance[valid_indices]
        
        # 移除NaN
        valid_mask2 = ~np.isnan(valid_abundance)
        valid_y2 = valid_y[valid_mask2]
        valid_abundance2 = valid_abundance[valid_mask2]
        
        if len(valid_y2) > 5 and np.std(valid_abundance2) > 1e-6 and np.std(valid_y2) > 1e-6:
            corr, pval = spearmanr(valid_abundance2, valid_y2)
            proximity_score = corr if not np.isnan(corr) else 0
        else:
            proximity_score = 0
        
        if verbose:
            print(f"      n={len(valid_y2)}, corr={proximity_score:.4f}")
        
        proximity_scores.append({
            'cell_type': ct,
            'proximity_score': proximity_score
        })
    
    prox_df = pd.DataFrame(proximity_scores)
    prox_df = prox_df.sort_values('proximity_score')
    prox_df.set_index('cell_type', inplace=True)
    
    if verbose:
        print(f"    Final prox_df shape: {prox_df.shape}")
    
    return prox_df

def permutation_test(df: pd.DataFrame,
                     cell_types: List[str],
                     cell_type_cols: List[str],
                     drug_col: str,
                     dist_col: str = 'interface_dist',
                     n_perm: int = 50,
                     radius: float = 0.1,
                     min_neighbors: int = 3,
                     verbose: bool = False,
                     random_seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    空间置换检验：重排细胞类型标签
    
    Parameters:
    -----------
    random_seed : int
        Random seed for reproducibility (default: 42)
    """
    # Set seed for reproducibility
    np.random.seed(random_seed)
    
    if verbose:
        print(f"  Random seed: {random_seed}")
        print("  Calculating real proximity scores...")
    
    real_prox_df = calculate_proximity_scores(
        df, cell_types, cell_type_cols, drug_col, dist_col, radius, min_neighbors, 
        verbose=verbose, random_seed=random_seed
    )
    
    if len(real_prox_df) == 0:
        if verbose:
            print("  No real proximity scores calculated")
        return real_prox_df, {}
    
    # 预计算坐标和邻居
    coords = df[['array_col', 'array_row']].values
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords_range = coords_max - coords_min
    if np.max(coords_range) > 0:
        coords = (coords - coords_min) / np.max(coords_range)
    
    tree = KDTree(coords)
    
    all_neighbors = []
    for i in range(len(df)):
        indices = tree.query_ball_point(coords[i], radius)
        indices = [idx for idx in indices if idx != i]
        all_neighbors.append(indices)
    
    valid_mask = [len(neighbors) >= min_neighbors for neighbors in all_neighbors]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    
    if len(valid_indices) < 10:
        if verbose:
            print("  Too few valid spots, returning zeros")
        for ct in real_prox_df.index:
            real_prox_df.loc[ct, 'p_value'] = 1.0
        return real_prox_df, {ct: [0] for ct in real_prox_df.index}
    
    if verbose:
        print(f"  Running {n_perm} permutations (shuffling cell types)...")
    
    perm_scores = {ct: [] for ct in real_prox_df.index}
    y = df[drug_col].values
    
    for i in range(n_perm):
        # Set seed for each permutation iteration for consistency
        perm_seed = random_seed + i
        np.random.seed(perm_seed)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"    Permutation {i+1}/{n_perm} (seed={perm_seed})")
        
        df_perm = df.copy()
        for ct_col in cell_type_cols:
            df_perm[ct_col] = np.random.permutation(df_perm[ct_col].values)
        
        for ct, ct_col in zip(cell_types, cell_type_cols):
            c = df_perm[ct_col].values
            neighbor_abundance = np.zeros(len(df))
            neighbor_abundance[:] = np.nan
            
            for j in valid_indices:
                neighbors = all_neighbors[j]
                if len(neighbors) > 0:
                    neighbor_abundance[j] = np.mean(c[neighbors])
            
            valid_y = y[valid_indices]
            valid_abundance = neighbor_abundance[valid_indices]
            valid_mask2 = ~np.isnan(valid_abundance)
            valid_y2 = valid_y[valid_mask2]
            valid_abundance2 = valid_abundance[valid_mask2]
            
            if len(valid_y2) > 5 and np.std(valid_abundance2) > 1e-6 and np.std(valid_y2) > 1e-6:
                corr, _ = spearmanr(valid_abundance2, valid_y2)
                score = corr if not np.isnan(corr) else 0
            else:
                score = 0
            
            if ct in perm_scores:
                perm_scores[ct].append(score)
    
    pvals = {}
    for ct in real_prox_df.index:
        real_score = real_prox_df.loc[ct, 'proximity_score']
        perm_vals = perm_scores.get(ct, [])
        
        if len(perm_vals) == 0:
            pvals[ct] = 1.0
        else:
            p = (np.sum(np.abs(perm_vals) >= abs(real_score)) + 1) / (len(perm_vals) + 1)
            pvals[ct] = p
    
    real_prox_df['p_value'] = real_prox_df.index.map(pvals)
    
    return real_prox_df, perm_scores

def plot_drug_gradient(bin_centers: np.ndarray,
                       drug_mean: np.ndarray,
                       title: str = "Drug spatial gradient",
                       xlabel: str = "Distance from tumor-stroma interface",
                       ylabel: str = "Normalized drug sensitivity",
                       figsize: Tuple[int, int] = (5, 4),
                       save: bool = False,
                       output_dir: str = "figure_outputs",
                       filename: str = "drug_gradient",
                       sample_id: str = "CRC1",
                       drug_name: str = "") -> None:
    plt.figure(figsize=figsize)
    plt.plot(bin_centers, drug_mean, linewidth=3, color='black')
    plt.axvline(0, linestyle="--", color='gray', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if save:
        os.makedirs(output_dir, exist_ok=True)
        if drug_name:
            full_filename = f"{sample_id}_{drug_name}_{filename}.pdf"
        else:
            full_filename = f"{sample_id}_{filename}.pdf"
        path = os.path.join(output_dir, full_filename)
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Figure saved: {path}")
    plt.show()

def plot_spatial_density(density_df: pd.DataFrame,
                         cell_types: List[str],
                         tumor_side_mean: Dict,
                         figsize: Tuple[float, float] = (6, None),
                         save: bool = False,
                         output_dir: str = "figure_outputs",
                         filename: str = "spatial_density",
                         sample_id: str = "CRC1",
                         drug_name: str = "") -> None:
    sorted_cell_types = sorted(cell_types, key=lambda x: tumor_side_mean[x], reverse=True)
    height_per_cell = 1.1
    if figsize[1] is None:
        figsize = (figsize[0], height_per_cell * len(sorted_cell_types))
    fig, axes = plt.subplots(len(sorted_cell_types), 1, figsize=figsize, sharex=True)
    if len(sorted_cell_types) == 1:
        axes = [axes]
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_cell_types)))
    for i, (ct, color) in enumerate(zip(sorted_cell_types, colors)):
        ax = axes[i]
        y = density_df[ct].values
        x = density_df['interface_dist'].values
        ax.fill_between(x, y, 0, color=color)
        ax.set_ylim(0, y.max())
        ax.margins(x=0, y=0)
        ax.text(
            x.min() + 0.02,
            y.max() * 0.7,
            ct,
            fontsize=12,
            weight='bold',
            ha='left',
            va='center'
        )
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color("black")
    axes[-1].set_xlabel("Distance from tumor/normal interface (scaled)", fontsize=10)
    plt.subplots_adjust(hspace=0)
    if save:
        os.makedirs(output_dir, exist_ok=True)
        if drug_name:
            full_filename = f"{sample_id}_{drug_name}_{filename}.pdf"
        else:
            full_filename = f"{sample_id}_{filename}.pdf"
        path = os.path.join(output_dir, full_filename)
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300, transparent=False)
        print(f"Figure saved: {path}")
    plt.show()

def plot_proximity_effect(prox_df: pd.DataFrame,
                          perm_scores: Dict,
                          figsize: Tuple[int, int] = (10, 6),
                          save: bool = False,
                          output_dir: str = "figure_outputs",
                          filename: str = "proximity_effect",
                          sample_id: str = "CRC1",
                          drug_name: str = "") -> None:
    if len(prox_df) == 0:
        print("Warning: No proximity scores to plot")
        return
    
    valid_cts = [ct for ct in prox_df.index if len(perm_scores.get(ct, [])) > 0]
    if len(valid_cts) == 0:
        print("Warning: No valid permutation scores found")
        # 即使没有perm_scores，也显示条形图
        valid_cts = prox_df.index.tolist()
    
    prox_df_filtered = prox_df.loc[valid_cts]
    
    if len(valid_cts) > 0 and len(perm_scores) > 0:
        perm_summary = pd.DataFrame({
            "median": [np.median(perm_scores.get(ct, [0])) for ct in valid_cts],
            "low": [np.percentile(perm_scores.get(ct, [-0.1, 0, 0.1]), 2.5) if len(perm_scores.get(ct, [])) > 0 else -0.1 for ct in valid_cts],
            "high": [np.percentile(perm_scores.get(ct, [-0.1, 0, 0.1]), 97.5) if len(perm_scores.get(ct, [])) > 0 else 0.1 for ct in valid_cts]
        }, index=valid_cts)
    else:
        perm_summary = pd.DataFrame({
            "median": [0] * len(valid_cts),
            "low": [-0.1] * len(valid_cts),
            "high": [0.1] * len(valid_cts)
        }, index=valid_cts)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    if drug_name:
        left_title = f"Observed cell proximity effect of {drug_name}"
    else:
        left_title = "Observed  cell proximity effect"
    
    axes[0].barh(
        prox_df_filtered.index,
        prox_df_filtered["proximity_score"],
        color=['#d95f5f' if v > 0 else '#5f8fd9' for v in prox_df_filtered["proximity_score"]]
    )
    axes[0].axvline(0, linestyle="--", color='gray')
    axes[0].set_title(left_title, fontsize=12)
    axes[0].set_xlabel("Cell proximity score")

    for i, (ct, row) in enumerate(prox_df_filtered.iterrows()):
        p = row.get("p_value", 1.0)
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            continue
        x = row["proximity_score"]
        if x > 0:
            axes[0].text(x + 0.005, i, star, va="center", fontsize=10)
        else:
            axes[0].text(x - 0.005, i, star, va="center", fontsize=10, ha='right')

    axes[1].errorbar(
        perm_summary["median"],
        perm_summary.index,
        xerr=[
            perm_summary["median"] - perm_summary["low"],
            perm_summary["high"] - perm_summary["median"]
        ],
        fmt="o",
        color="black",
        ecolor="gray",
        capsize=3
    )
    axes[1].axvline(0, linestyle="--", color='gray')
    axes[1].set_title("Spatially permuted", fontsize=12)
    axes[1].set_xlabel("Median proximity score (95% CI)")
    axes[1].set_xlim(-1, 1)
    plt.tight_layout()
    if save:
        os.makedirs(output_dir, exist_ok=True)
        if drug_name:
            full_filename = f"{sample_id}_{drug_name}_{filename}.pdf"
        else:
            full_filename = f"{sample_id}_{filename}.pdf"
        path = os.path.join(output_dir, full_filename)
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Figure saved: {path}")
    plt.show()

def analyze_spatial_proximity_effect(adata,
                                     drug_name: str = "Irinotecan",
                                     sample_id: str = "CRC1",
                                     cell_type_prefix: str = "q95_spot_factors",
                                     region_col: str = "Region_interface",
                                     row_col: str = "array_row",
                                     col_col: str = "array_col",
                                     n_perm: int = 50,
                                     n_bins: int = 80,
                                     smooth_sigma: float = 2.0,
                                     save_figures: bool = True,
                                     output_dir: str = "figure_outputs",
                                     figsize_density: Tuple[float, float] = (6, None),
                                     use_raw_obs: bool = False,
                                     verbose: bool = True,
                                     radius: float = 0.1,
                                     min_neighbors: int = 3,
                                     random_seed: int = 42) -> Dict:
    """
    Main analysis function with random seed control.
    
    Parameters:
    -----------
    random_seed : int
        Random seed for reproducibility throughout the analysis (default: 42)
    """
    # Set random seeds at the beginning of analysis
    set_random_seeds(random_seed)
    
    if verbose:
        print("=" * 60)
        print(f"Starting spatial proximity effect analysis for {sample_id}")
        print(f"Random seed: {random_seed}")
        print("=" * 60)
    
    if verbose:
        print("\nStep 0: Extracting obs data from AnnData...")
    if use_raw_obs and hasattr(adata, 'raw') and hasattr(adata.raw, 'obs'):
        df = adata.raw.obs.copy()
    else:
        df = extract_obs_from_adata(adata)
    if verbose:
        print(f"  Data shape: {df.shape}")
    
    if verbose:
        print("\nStep 1-3: Calculating interface distance...")
    df = calculate_interface_distance(df, row_col, col_col, region_col)
    df = normalize_distance(df)
    
    if verbose:
        print(f"  Distance range: {df['interface_dist'].min():.4f} to {df['interface_dist'].max():.4f}")
        print(f"  Positive distances (tumor): {(df['interface_dist'] > 0).sum()}")
        print(f"  Negative distances (stroma): {(df['interface_dist'] < 0).sum()}")
    
    if verbose:
        print("\nStep 4: Extracting cell types...")
    cell_types, cell_type_cols = extract_cell_types(df, prefix=cell_type_prefix)
    if verbose:
        print(f"  Found {len(cell_types)} cell types")
        if len(cell_types) > 0:
            print(f"  First few: {cell_types[:5]}")
    
    if verbose:
        print("\nStep 5-6: Calculating spatial density...")
    density_df, tumor_side_mean = calculate_spatial_density(
        df, cell_types, cell_type_cols, 
        dist_col='interface_dist',
        n_bins=n_bins, 
        smooth_sigma=smooth_sigma
    )
    
    if verbose:
        print("\nStep 7: Calculating drug gradient...")
    drug_col = get_drug_column_name(adata, drug_name, sample_id)
    if verbose:
        print(f"  Using drug column: {drug_col}")
        if drug_col in df.columns:
            print(f"  Drug values range: {df[drug_col].min():.4f} to {df[drug_col].max():.4f}")
    
    bin_centers, drug_mean = calculate_drug_gradient(
        df, drug_col, 
        dist_col='interface_dist',
        n_bins=n_bins, 
        smooth_sigma=smooth_sigma
    )
    
    if verbose:
        print("\nStep 8-9: Calculating proximity scores and permutation test...")
        print(f"  Parameters: radius={radius}, min_neighbors={min_neighbors}, n_perm={n_perm}")
    
    prox_df, perm_scores = permutation_test(
        df, cell_types, cell_type_cols, drug_col, 
        dist_col='interface_dist',
        n_perm=n_perm,
        radius=radius,
        min_neighbors=min_neighbors,
        verbose=verbose,
        random_seed=random_seed
    )
    
    sig_cells = prox_df[prox_df["p_value"] < 0.05] if len(prox_df) > 0 else pd.DataFrame()
    if verbose:
        print(f"\nSignificant proximity cells (p < 0.05): {len(sig_cells)}")
        if len(sig_cells) > 0:
            print(sig_cells)
    
    if verbose:
        print("\nGenerating figures...")
    
    plot_drug_gradient(
        bin_centers, drug_mean,
        save=save_figures, 
        output_dir=output_dir, 
        filename="drug_gradient",
        sample_id=sample_id,
        drug_name=drug_name
    )
    
    plot_spatial_density(
        density_df, cell_types, tumor_side_mean,
        figsize=figsize_density,
        save=save_figures, 
        output_dir=output_dir, 
        filename="spatial_density",
        sample_id=sample_id,
        drug_name=drug_name
    )
    
    plot_proximity_effect(
        prox_df, perm_scores,
        save=save_figures, 
        output_dir=output_dir, 
        filename="proximity_effect",
        sample_id=sample_id,
        drug_name=drug_name
    )
    
    results = {
        'data': df,
        'cell_types': cell_types,
        'cell_type_cols': cell_type_cols,
        'density_df': density_df,
        'tumor_side_mean': tumor_side_mean,
        'bin_centers': bin_centers,
        'drug_gradient': drug_mean,
        'proximity_scores': prox_df,
        'permutation_scores': perm_scores,
        'significant_cells': sig_cells,
        'sample_id': sample_id,
        'drug_col': drug_col,
        'drug_name': drug_name,
        'random_seed': random_seed
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("=" * 60)
    
    return results

def batch_analyze_spatial_proximity(adata_dict: Dict[str, Any],
                                    drug_name: str = "Irinotecan",
                                    random_seed: int = 42,
                                    **kwargs) -> Dict[str, Dict]:
    """
    Batch analysis with consistent random seed across samples.
    
    Parameters:
    -----------
    random_seed : int
        Base random seed for reproducibility (default: 42)
        Each sample will use a different seed: random_seed + sample_index
    """
    results = {}
    for i, (sample_id, adata) in enumerate(adata_dict.items()):
        # Use different seed for each sample for independence
        sample_seed = random_seed + i
        
        print(f"\nProcessing sample: {sample_id}")
        print(f"Using random seed: {sample_seed}")
        print("-" * 40)
        
        try:
            result = analyze_spatial_proximity_effect(
                adata, 
                drug_name=drug_name,
                sample_id=sample_id,
                random_seed=sample_seed,
                **kwargs
            )
            results[sample_id] = result
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    return results

def summarize_results(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    summary_list = []
    for sample_id, results in results_dict.items():
        sig_cells = results['significant_cells']
        if len(sig_cells) > 0:
            for ct, row in sig_cells.iterrows():
                summary_list.append({
                    'sample': sample_id,
                    'cell_type': ct,
                    'proximity_score': row['proximity_score'],
                    'p_value': row['p_value']
                })
    if summary_list:
        return pd.DataFrame(summary_list).sort_values(['sample', 'p_value'])
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage with random seed
    # set_random_seeds(42)  # Already set at module level
    
    # For custom seed in analysis:
    # results = analyze_spatial_proximity_effect(
    #     adata, 
    #     drug_name="Irinotecan",
    #     sample_id="CRC1",
    #     random_seed=123,  # Custom seed
    #     verbose=True
    # )
    
    pass