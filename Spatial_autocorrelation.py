import pandas as pd
import scanpy as sc
from libpysal.weights import KNN
from esda import Moran
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams

rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
sensitivity_colors = {
    'Sensitive': '#FF0000',
    'Resistant': '#0072B2',
    'Unknown': '#808080'
}

def classify_sensitivity(value):
    if value > 0.70:
        return 'Sensitive'
    elif value < 0.30:
        return 'Resistant'
    elif 0.30 <= value <= 0.70:
        return 'Uncertain'
    else:
        return 'Unknown'

def analyze_and_plot_spatial_autocorrelation(adata_path, drugs, cancer_sample_name, titles=None, k=6, save_folder=None):
    
    if save_folder is None:
        save_folder = f"{cancer_sample_name}_Result"
    spatial_folder = os.path.join(save_folder, f"{cancer_sample_name}_SpatialAutocorrelation")
    os.makedirs(spatial_folder, exist_ok=True)
    
    print("="*60)
    print("STEP 1: Loading AnnData object")
    print("="*60)
    try:
        adata = sc.read(adata_path)
        print(f"✓ Successfully loaded AnnData with {adata.n_obs} cells")
        print(f"✓ Spatial coordinates available: {'spatial' in adata.obsm.keys()}")
    except Exception as e:
        print(f"✗ Error loading AnnData: {e}")
        return None
    
    print("\n" + "="*60)
    print("STEP 2: Calculating spatial autocorrelation (Moran's I)")
    print("="*60)
    
    cell_types = ['Sensitive', 'Resistant', 'Unknown']
    results = []
    
    for drug in drugs:
        sensitivity_col = f"{drug}_Drug_Sensitivity_Predictions"
        
        if sensitivity_col not in adata.obs.columns:
            print(f"⚠ Warning: Column {sensitivity_col} not found. Skipping {drug}.")
            continue
            
        sensitivity_class_col = f"{drug}_Sensitivity"
        adata.obs[sensitivity_class_col] = adata.obs[sensitivity_col].apply(classify_sensitivity)
        
        print(f"\nProcessing drug: {drug}")
        
        for cell_type in cell_types:
            cell_type_indices = adata.obs[adata.obs[sensitivity_class_col] == cell_type].index
            cell_type_data = adata.obs.loc[cell_type_indices, sensitivity_col]
            
            if cell_type_data.empty:
                print(f"  - {cell_type}: No data")
                results.append({
                    "Drug": drug,
                    "CellType": cell_type,
                    "Moran_I": np.nan,
                    "p_value": np.nan,
                    "n_cells": 0
                })
                continue
            
            cell_type_indices_loc = [adata.obs.index.get_loc(idx) for idx in cell_type_indices]
            cell_type_coords = adata.obsm['spatial'][cell_type_indices_loc]
            
            if np.isnan(cell_type_coords).any() or np.isinf(cell_type_coords).any():
                print(f"  - {cell_type}: Invalid coordinates")
                results.append({
                    "Drug": drug,
                    "CellType": cell_type,
                    "Moran_I": np.nan,
                    "p_value": np.nan,
                    "n_cells": len(cell_type_data)
                })
                continue
            
            try:
                if len(cell_type_coords) < 2:
                    print(f"  - {cell_type}: Insufficient cells (n={len(cell_type_coords)})")
                    results.append({
                        "Drug": drug,
                        "CellType": cell_type,
                        "Moran_I": np.nan,
                        "p_value": np.nan,
                        "n_cells": len(cell_type_data)
                    })
                    continue
                
                actual_k = min(k, len(cell_type_coords) - 1)
                cell_type_weights = KNN.from_array(cell_type_coords, k=actual_k)
                moran = Moran(cell_type_data, cell_type_weights)
                
                significance = "***" if moran.p_sim < 0.001 else "**" if moran.p_sim < 0.01 else "*" if moran.p_sim < 0.05 else ""
                print(f"  - {cell_type}: Moran's I = {moran.I:.4f}, p-value = {moran.p_sim:.4f} {significance}, n = {len(cell_type_data)}")
                
                results.append({
                    "Drug": drug,
                    "CellType": cell_type,
                    "Moran_I": moran.I,
                    "p_value": moran.p_sim,
                    "n_cells": len(cell_type_data)
                })
                
            except Exception as e:
                print(f"  - {cell_type}: Error - {e}")
                results.append({
                    "Drug": drug,
                    "CellType": cell_type,
                    "Moran_I": np.nan,
                    "p_value": np.nan,
                    "n_cells": len(cell_type_data)
                })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("STEP 2.5: FDR Multiple Testing Correction")
    print("="*60)
    
    valid_mask = (results_df['CellType'].isin(['Sensitive', 'Resistant'])) & \
                 (results_df['p_value'].notna()) & \
                 (~np.isinf(results_df['p_value']))
    
    valid_p_values = results_df.loc[valid_mask, 'p_value'].values
    
    if len(valid_p_values) > 0:
        rejected, p_adjusted, _, _ = multipletests(valid_p_values, alpha=0.05, method='fdr_bh')
        results_df['p_value_fdr'] = np.nan
        results_df.loc[valid_mask, 'p_value_fdr'] = p_adjusted
        
        n_tests = len(valid_p_values)
        n_significant_raw = np.sum(valid_p_values < 0.05)
        n_significant_fdr = np.sum(p_adjusted < 0.05)
        
        print(f"\n FDR Correction Summary:")
        print(f"  - Number of tests: {n_tests}")
        print(f"  - Significant (raw p < 0.05): {n_significant_raw}")
        print(f"  - Significant (FDR p < 0.05): {n_significant_fdr}")
    else:
        results_df['p_value_fdr'] = np.nan
        print("\n⚠ No valid p-values found for FDR correction")
    
    csv_path = os.path.join(spatial_folder, f"{cancer_sample_name}_spatial_autocorrelation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")
    
    print("\n" + "="*60)
    print("STEP 3: Plotting results for each drug")
    print("="*60)
    
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })
    
    plot_folder = os.path.join(spatial_folder, "plots")
    os.makedirs(plot_folder, exist_ok=True)
    
    if titles is not None and len(titles) == len(drugs):
        title_dict = dict(zip(drugs, titles))
    else:
        title_dict = {drug: drug for drug in drugs}
    
    drugs_with_data = results_df['Drug'].unique()
    for drug in drugs_with_data:
        drug_df = results_df[results_df["Drug"] == drug].copy()
        drug_df = drug_df[drug_df["CellType"].isin(['Sensitive', 'Resistant'])]
        drug_df = drug_df.dropna(subset=["CellType", "Moran_I"])
        
        if drug_df.empty:
            print(f"  - {drug}: No valid data for plotting")
            continue
        
        category_order = ['Sensitive', 'Resistant']
        drug_df['CellType'] = pd.Categorical(drug_df['CellType'], categories=category_order, ordered=True)
        drug_df = drug_df.sort_values('CellType')
        
        fig, ax = plt.subplots(figsize=(6, 3))
        
        bars = sns.barplot(
            data=drug_df,
            x="Moran_I",
            y="CellType",
            hue="CellType",
            orient="h",
            palette=sensitivity_colors,
            dodge=False,
            ax=ax,
            legend=False
        )
        
        for i, (_, row) in enumerate(drug_df.iterrows()):
            p_val_to_use = row['p_value_fdr'] if pd.notna(row['p_value_fdr']) else row['p_value']
            if p_val_to_use < 0.05:
                x_pos = row['Moran_I'] + 0.02 if row['Moran_I'] >= 0 else row['Moran_I'] - 0.05
                ax.text(x_pos, i, '*', fontsize=16, va='center', ha='center')
        
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
        
        plot_title = title_dict.get(drug, drug)
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Moran's I value", fontsize=14, fontweight='bold')
        ax.set_ylabel("", fontsize=14, fontweight='bold')
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        
        plot_path = os.path.join(plot_folder, f"{drug}_spatial_autocorrelation.png")
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        
        # Force display in Jupyter Notebook
        plt.show(block=True)
        
        print(f"  - {drug}: Plot saved to {plot_path}")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    total_drugs = len(drugs_with_data)
    total_calculations = len(results_df.dropna(subset=['Moran_I']))
    
    print(f"\n Summary Statistics:")
    print(f"  - Total drugs analyzed: {total_drugs}")
    print(f"  - Total calculations: {total_calculations}")
    
    significant_results = results_df[(results_df['p_value_fdr'] < 0.05) & (results_df['p_value_fdr'].notna())].dropna(subset=['Moran_I'])
    significant_results = significant_results[significant_results['CellType'].isin(['Sensitive', 'Resistant'])]
    
    if not significant_results.empty:
        print(f"\n Significant results (FDR-adjusted p < 0.05):")
        sig_summary = significant_results.groupby('Drug').size()
        for drug, count in sig_summary.items():
            display_name = title_dict.get(drug, drug)
            print(f"  - {display_name}: {count} significant categories")
            
        print(f"\n  By category:")
        for category in ['Sensitive', 'Resistant']:
            cat_sig = significant_results[significant_results['CellType'] == category]
            if not cat_sig.empty:
                for _, row in cat_sig.iterrows():
                    display_name = title_dict.get(row['Drug'], row['Drug'])
                    print(f"    - {display_name}: {category} - Moran's I = {row['Moran_I']:.4f}")
    else:
        print("\n No significant results found (FDR-adjusted p < 0.05)")
    
    valid_results = results_df.dropna(subset=['Moran_I'])
    valid_results = valid_results[valid_results['CellType'].isin(['Sensitive', 'Resistant'])]
    
    if not valid_results.empty:
        print(f"\n Top 3 positive Moran's I:")
        top_positive = valid_results.nlargest(3, 'Moran_I')[['Drug', 'CellType', 'Moran_I', 'p_value_fdr']]
        for _, row in top_positive.iterrows():
            display_name = title_dict.get(row['Drug'], row['Drug'])
            fdr_sig = " (FDR significant)" if pd.notna(row['p_value_fdr']) and row['p_value_fdr'] < 0.05 else ""
            print(f"  - {display_name} - {row['CellType']}: {row['Moran_I']:.4f}{fdr_sig}")
        
        print(f"\n Top 3 negative Moran's I:")
        top_negative = valid_results.nsmallest(3, 'Moran_I')[['Drug', 'CellType', 'Moran_I', 'p_value_fdr']]
        for _, row in top_negative.iterrows():
            display_name = title_dict.get(row['Drug'], row['Drug'])
            fdr_sig = " (FDR significant)" if pd.notna(row['p_value_fdr']) and row['p_value_fdr'] < 0.05 else ""
            print(f"  - {display_name} - {row['CellType']}: {row['Moran_I']:.4f}{fdr_sig}")
    
    print(f"\n Results saved in: {spatial_folder}")
    print(f"   - CSV file: {csv_path}")
    print(f"   - Plots folder: {plot_folder}")
    
    return results_df