import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from scipy.stats import spearmanr
import os
from typing import Optional, List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import AnnData, use Any type if failed
try:
    from anndata import AnnData
except ImportError:
    AnnData = Any
    print("Warning: anndata not installed, using Any type for AnnData")

# ===============================
# AnnData Processing Module
# ===============================

def extract_obs_from_adata(adata, use_raw: bool = False) -> pd.DataFrame:
    """
    Extract obs data from AnnData object
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object
    use_raw : bool
        Whether to use information from raw data
    
    Returns:
    --------
    pd.DataFrame : obs dataframe
    """
    if hasattr(adata, 'obs'):
        return adata.obs.copy()
    else:
        raise ValueError("Input object does not have obs attribute")


def get_drug_column_name(adata, drug_name: str, sample_id: str = "CRC1") -> str:
    """
    Get drug column name
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object
    drug_name : str
        Drug name
    sample_id : str
        Sample ID
    
    Returns:
    --------
    str : Complete drug column name
    """
    full_name = f"{sample_id}_{drug_name}"
    
    # Check if column exists
    if hasattr(adata, 'obs') and full_name in adata.obs.columns:
        return full_name
    else:
        # Try to find columns containing drug name
        if hasattr(adata, 'obs'):
            for col in adata.obs.columns:
                if drug_name.lower() in col.lower():
                    return col
        return full_name


# ===============================
# Core Analysis Functions Module
# ===============================

def calculate_interface_distance(df: pd.DataFrame, 
                                 row_col: str = 'array_row', 
                                 col_col: str = 'array_col',
                                 region_col: str = 'Region_interface',
                                 interface_label: str = 'Interface',
                                 stroma_label: str = 'Stroma') -> pd.DataFrame:
    """
    Calculate distance from each spot to tumor-stroma interface
    """
    df = df.copy()
    df[row_col] = df[row_col].astype(int)
    df[col_col] = df[col_col].astype(int)
    
    rows = df[row_col].max() + 1
    cols = df[col_col].max() + 1
    
    # Create binary grid (True for tissue regions, False for interface)
    grid = np.ones((rows, cols), dtype=bool)
    for r, c, region in zip(df[row_col], df[col_col], df[region_col]):
        if region == interface_label:
            grid[r, c] = False
    
    # Calculate distance transform
    dist = distance_transform_edt(grid)
    
    # Assign distance to each spot (stroma side as negative values)
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
    """
    Normalize interface distance to [-1, 1] range
    """
    df = df.copy()
    max_abs = np.max(np.abs(df[dist_col]))
    if max_abs > 0:
        df[dist_col] = df[dist_col] / max_abs
    return df


def extract_cell_types(df: pd.DataFrame, 
                       prefix: str = 'q95_spot_factors') -> Tuple[List[str], List[str]]:
    """
    Extract cell type column names and corresponding cell type names
    """
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
    """
    Calculate spatial density distribution of cell types along interface distance
    """
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
    """
    Calculate drug sensitivity gradient along the interface
    """
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
                               dist_col: str = 'interface_dist') -> pd.DataFrame:
    """
    Calculate proximity effect score for each cell type
    """
    tumor_df = df[df[dist_col] > 0]
    stroma_df = df[df[dist_col] < 0]
    
    prox_scores = {}
    
    for ct, col in zip(cell_types, cell_type_cols):
        # Avoid NaN in Spearman correlation calculation
        if tumor_df[col].std() == 0 or stroma_df[col].std() == 0:
            prox_scores[ct] = 0
            continue
        
        r_tumor, _ = spearmanr(tumor_df[col], tumor_df[drug_col])
        r_stroma, _ = spearmanr(stroma_df[col], stroma_df[drug_col])
        
        r_tumor = 0 if np.isnan(r_tumor) else r_tumor
        r_stroma = 0 if np.isnan(r_stroma) else r_stroma
        
        prox_scores[ct] = r_tumor - r_stroma
    
    prox_df = pd.DataFrame.from_dict(prox_scores, orient="index", columns=["proximity_score"])
    prox_df = prox_df.sort_values("proximity_score")
    
    return prox_df


def permutation_test(df: pd.DataFrame,
                     cell_types: List[str],
                     cell_type_cols: List[str],
                     drug_col: str,
                     dist_col: str = 'interface_dist',
                     n_perm: int = 50) -> Tuple[pd.DataFrame, Dict]:
    """
    Permutation test to assess significance
    """
    # First calculate real proximity scores
    prox_df = calculate_proximity_scores(
        df, cell_types, cell_type_cols, drug_col, dist_col
    )

    perm_scores = {ct: [] for ct in cell_types}

    for i in range(n_perm):
        shuffled = np.random.permutation(df[drug_col].values)
        df["drug_perm"] = shuffled

        tumor_perm = df[df[dist_col] > 0]
        stroma_perm = df[df[dist_col] < 0]

        for ct, col in zip(cell_types, cell_type_cols):
            if tumor_perm[col].std() == 0 or stroma_perm[col].std() == 0:
                perm_scores[ct].append(0)
                continue

            r_t, _ = spearmanr(tumor_perm[col], tumor_perm["drug_perm"])
            r_s, _ = spearmanr(stroma_perm[col], stroma_perm["drug_perm"])

            r_t = 0 if np.isnan(r_t) else r_t
            r_s = 0 if np.isnan(r_s) else r_s

            perm_scores[ct].append(r_t - r_s)

    # Calculate p-values
    pvals = {}
    for ct in cell_types:
        real = prox_df.loc[ct, "proximity_score"]
        perm = np.array(perm_scores[ct])
        p = (np.sum(np.abs(perm) >= abs(real)) + 1) / (len(perm) + 1)
        pvals[ct] = p

    prox_df["p_value"] = prox_df.index.map(pvals)
    
    # Remove temporary column
    if "drug_perm" in df.columns:
        df.drop("drug_perm", axis=1, inplace=True)

    return prox_df, perm_scores


# ===============================
# Visualization Functions Module
# ===============================

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
    """
    Plot drug spatial gradient
    """
    plt.figure(figsize=figsize)
    plt.plot(bin_centers, drug_mean, linewidth=3, color='black')
    plt.axvline(0, linestyle="--", color='gray', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        # Add drug name to filename
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
    """
    Plot cell spatial density distribution (following user's provided style)
    """
    # Sort by tumor side mean
    sorted_cell_types = sorted(cell_types, key=lambda x: tumor_side_mean[x], reverse=True)
    
    # Calculate figure height
    height_per_cell = 1.1
    if figsize[1] is None:
        figsize = (figsize[0], height_per_cell * len(sorted_cell_types))
    
    # Create subplots
    fig, axes = plt.subplots(len(sorted_cell_types), 1, figsize=figsize, sharex=True)
    
    # If only one cell type, ensure axes is indexable
    if len(sorted_cell_types) == 1:
        axes = [axes]
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_cell_types)))
    
    for i, (ct, color) in enumerate(zip(sorted_cell_types, colors)):
        ax = axes[i]
        y = density_df[ct].values
        x = density_df['interface_dist'].values
        
        # Fill area
        ax.fill_between(x, y, 0, color=color)
        
        # Force tight margins
        ax.set_ylim(0, y.max())
        ax.margins(x=0, y=0)
        
        # Label inside the plot
        ax.text(
            x.min() + 0.02,
            y.max() * 0.7,
            ct,
            fontsize=12,
            weight='bold',
            ha='left',
            va='center'
        )
        
        # Remove ticks
        ax.set_yticks([])
        
        # Black borders
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color("black")
    
    # X-axis label
    axes[-1].set_xlabel("Distance from tumor/normal interface (scaled)", fontsize=10)
    
    # Tight panel spacing
    plt.subplots_adjust(hspace=0)
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        # Add drug name to filename
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
    """
    Plot proximity effect dual-panel figure
    
    Parameters:
    -----------
    prox_df : pd.DataFrame
        Proximity effect scores dataframe
    perm_scores : Dict
        Permutation test results
    figsize : Tuple[int, int]
        Figure size
    save : bool
        Whether to save figure
    output_dir : str
        Output directory
    filename : str
        Filename prefix
    sample_id : str
        Sample ID
    drug_name : str
        Drug name for title
    """
    perm_summary = pd.DataFrame({
        "median": [np.median(perm_scores[ct]) for ct in prox_df.index],
        "low": [np.percentile(perm_scores[ct], 2.5) for ct in prox_df.index],
        "high": [np.percentile(perm_scores[ct], 97.5) for ct in prox_df.index]
    }, index=prox_df.index)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Left panel - add drug name to title
    if drug_name:
        left_title = f"Observed proximity effect of {drug_name}"
    else:
        left_title = "Observed proximity effect"
    
    bars = axes[0].barh(
        prox_df.index,
        prox_df["proximity_score"],
        color=['#d95f5f' if v > 0 else '#5f8fd9' for v in prox_df["proximity_score"]]
    )

    axes[0].axvline(0, linestyle="--", color='gray')
    axes[0].set_title(left_title, fontsize=12)
    axes[0].set_xlabel("Proximity score")

    # Significance stars - placed close to bars
    for i, (ct, row) in enumerate(prox_df.iterrows()):
        p = row["p_value"]
        
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
            # Positive values: stars on the right side, flush with bar edge
            axes[0].text(x + 0.005, i, star, va="center", fontsize=10)
        else:
            # Negative values: stars on the left side, flush with bar edge
            axes[0].text(x - 0.005, i, star, va="center", fontsize=10, ha='right')

    # Right panel
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

    # Fix narrow range
    max_abs = max(abs(prox_df["proximity_score"].min()), 
                  abs(prox_df["proximity_score"].max()))
    axes[1].set_xlim(-max_abs*0.8, max_abs*0.8)

    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        # Add drug name to filename
        if drug_name:
            full_filename = f"{sample_id}_{drug_name}_{filename}.pdf"
        else:
            full_filename = f"{sample_id}_{filename}.pdf"
        path = os.path.join(output_dir, full_filename)
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Figure saved: {path}")

    plt.show()

# ===============================
# Main Analysis Function
# ===============================

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
                                     verbose: bool = True) -> Dict:
    """
    Main analysis function: Integrates all steps to analyze spatial proximity effect
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing obs data
    drug_name : str
        Drug name
    sample_id : str
        Sample ID
    cell_type_prefix : str
        Prefix for cell type columns
    region_col : str
        Column name for region information
    row_col, col_col : str
        Column names for row and column coordinates
    n_perm : int
        Number of permutations
    n_bins : int
        Number of distance bins
    smooth_sigma : float
        Gaussian smoothing parameter
    save_figures : bool
        Whether to save figures
    output_dir : str
        Output directory
    figsize_density : Tuple[float, float]
        Density figure size
    use_raw_obs : bool
        Whether to use raw.obs
    verbose : bool
        Whether to print detailed information
    
    Returns:
    --------
    Dict : Dictionary containing all analysis results
    """
    if verbose:
        print("=" * 60)
        print(f"Starting spatial proximity effect analysis for {sample_id}")
        print("=" * 60)
    
    # Step 0: Extract obs data
    if verbose:
        print("\nStep 0: Extracting obs data from AnnData...")
    
    if use_raw_obs and hasattr(adata, 'raw') and hasattr(adata.raw, 'obs'):
        df = adata.raw.obs.copy()
    else:
        df = extract_obs_from_adata(adata)
    
    if verbose:
        print(f"  Data shape: {df.shape}")
    
    # Steps 1-3: Calculate and normalize interface distance
    if verbose:
        print("\nSteps 1-3: Calculating interface distance...")
    df = calculate_interface_distance(
        df, 
        row_col=row_col, 
        col_col=col_col,
        region_col=region_col
    )
    df = normalize_distance(df)
    
    # Step 4: Extract cell types
    if verbose:
        print("\nStep 4: Extracting cell types...")
    cell_types, cell_type_cols = extract_cell_types(df, prefix=cell_type_prefix)
    if verbose:
        print(f"  Found {len(cell_types)} cell types")
        if len(cell_types) > 0:
            print(f"  First few: {cell_types[:5]}")
    
    # Steps 5-6: Calculate spatial density
    if verbose:
        print("\nSteps 5-6: Calculating spatial density...")
    density_df, tumor_side_mean = calculate_spatial_density(
        df, cell_types, cell_type_cols, 
        dist_col='interface_dist',
        n_bins=n_bins, 
        smooth_sigma=smooth_sigma
    )
    
    # Step 7: Calculate drug gradient
    if verbose:
        print("\nStep 7: Calculating drug gradient...")
    drug_col = get_drug_column_name(adata, drug_name, sample_id)
    if verbose:
        print(f"  Using drug column: {drug_col}")
    
    bin_centers, drug_mean = calculate_drug_gradient(
        df, drug_col, 
        dist_col='interface_dist',
        n_bins=n_bins, 
        smooth_sigma=smooth_sigma
    )
    
    # Steps 8-9: Calculate proximity scores and permutation test
    if verbose:
        print("\nSteps 8-9: Calculating proximity scores and permutation test...")
    prox_df, perm_scores = permutation_test(
        df, cell_types, cell_type_cols, drug_col, 
        dist_col='interface_dist',
        n_perm=n_perm
    )
    
    # Find significant cell types
    sig_cells = prox_df[prox_df["p_value"] < 0.05]
    if verbose:
        print(f"\nSignificant proximity cells (p < 0.05): {len(sig_cells)}")
        if len(sig_cells) > 0:
            print(sig_cells)
    
    # Generate figures
    if verbose:
        print("\nGenerating figures...")
    
    # Figure 1: Drug gradient - pass drug_name
    plot_drug_gradient(
        bin_centers, drug_mean,
        save=save_figures, 
        output_dir=output_dir, 
        filename="drug_gradient",
        sample_id=sample_id,
        drug_name=drug_name
    )
    
    # Figure 2: Cell spatial density - pass drug_name
    plot_spatial_density(
        density_df, cell_types, tumor_side_mean,
        figsize=figsize_density,
        save=save_figures, 
        output_dir=output_dir, 
        filename="spatial_density",
        sample_id=sample_id,
        drug_name=drug_name
    )
    
    # Figure 3: Proximity effect - pass drug_name
    plot_proximity_effect(
        prox_df, perm_scores,
        save=save_figures, 
        output_dir=output_dir, 
        filename="proximity_effect",
        sample_id=sample_id,
        drug_name=drug_name
    )
    
    # Return all results
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
        'drug_name': drug_name
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("=" * 60)
    
    return results


# ===============================
# Batch Analysis Function
# ===============================

def batch_analyze_spatial_proximity(adata_dict: Dict[str, Any],
                                    drug_name: str = "Irinotecan",
                                    **kwargs) -> Dict[str, Dict]:
    """
    Batch analyze multiple samples
    
    Parameters:
    -----------
    adata_dict : Dict[str, AnnData]
        Dictionary mapping sample IDs to AnnData objects
    drug_name : str
        Drug name
    **kwargs : 
        Additional parameters to pass to analyze_spatial_proximity_effect
    
    Returns:
    --------
    Dict[str, Dict] : Analysis results for each sample
    """
    results = {}
    
    for sample_id, adata in adata_dict.items():
        print(f"\nProcessing sample: {sample_id}")
        print("-" * 40)
        
        try:
            result = analyze_spatial_proximity_effect(
                adata, 
                drug_name=drug_name,
                sample_id=sample_id,
                **kwargs
            )
            results[sample_id] = result
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


# ===============================
# Results Summary Function
# ===============================

def summarize_results(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Summarize analysis results from multiple samples
    
    Parameters:
    -----------
    results_dict : Dict[str, Dict]
        Results dictionary from batch analysis
    
    Returns:
    --------
    pd.DataFrame : Summary table of significant cell types
    """
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


# ===============================
# Usage Examples
# ===============================

if __name__ == "__main__":
    # Single sample analysis example
    # results = analyze_spatial_proximity_effect(
    #     adata=CRC1,  # Assuming CRC1 is an AnnData object
    #     drug_name="Irinotecan",
    #     sample_id="CRC1",
    #     n_perm=50,
    #     save_figures=True,
    #     output_dir="figure_outputs",
    #     verbose=True
    # )
    
    # Batch analysis example
    # adata_dict = {
    #     "CRC1": CRC1,
    #     "CRC2": CRC2,
    #     "CRC3": CRC3
    # }
    # batch_results = batch_analyze_spatial_proximity(
    #     adata_dict,
    #     drug_name="Irinotecan",
    #     n_perm=50,
    #     save_figures=True
    # )
    
    # Summarize results
    # summary_df = summarize_results(batch_results)
    # print(summary_df)
    
    # Access individual results
    # prox_scores = results['proximity_scores']
    # sig_cells = results['significant_cells']
    # density_data = results['density_df']
    pass