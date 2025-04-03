import pandas as pd
import scanpy as sc
from libpysal.weights import KNN
from esda import Moran
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import scipy.stats as stats

# Functions to load files
def load_files(adata_path, deconv_path, interface_path):
    """
    Load the data files.
    :param adata_path: Path to AnnData file
    :param deconv_path: Path to deconvolution data file
    :param interface_path: Path to region data file
    :return: adata, deconv_data, interface_data
    """
    adata = sc.read(adata_path)
    deconv_data = pd.read_csv(deconv_path, header=None).T
    interface_data = pd.read_csv(interface_path)
    return adata, deconv_data, interface_data

# Functions that work with deconvoluted data
def process_deconvolution_data(deconv_data, adata_obs_index):
    """
    Process the deconvoluted data and convert it into a DataFrame that matches adata.obs.
    :param deconv_data: Deconvolution data
    :param adata_obs_index: Index of adata.obs
    :return: deconvolution_df
    """
    cell_codes = deconv_data.iloc[0, 1:].values
    data_values = deconv_data.iloc[1:, 1:].values
    deconvolution_df = pd.DataFrame(data_values, columns=cell_codes).astype(float)
    deconvolution_df.index = deconv_data.iloc[1:, 0].values
    deconvolution_df = deconvolution_df.reindex(adata_obs_index)
    return deconvolution_df

# Update the function of adata.obs
def update_obs_with_deconvolution(adata, deconv_df, columns_to_convert):
    """
    Add the deconvoluted data to adata.obs.
    :param adata: AnnData object
    :param deconv_df: Deconvolution data DataFrame
    :param columns_to_convert: Columns that need to be converted to float
    :return: adata
    """
    adata.obs = pd.concat([adata.obs, deconv_df], axis=1)
    for col in columns_to_convert:
        adata.obs[col] = adata.obs[col].astype(float)
    return adata

# Categorical drug susceptibility level as a function of the level
def classify_sensitivity(value):
    """
    Classify based on drug sensitivity values.
    :param value: Drug sensitivity value
    :return: Classification result
    """
    if 0.80 <= value <= 1.0:
        return 'High sensitive'
    elif 0.60 <= value < 0.80:
        return 'Low sensitive'
    elif 0.40 < value <= 0.60:
        return 'Uncertain'
    elif 0.0 <= value <= 0.20:
        return 'High resistant'
    elif 0.20 < value <= 0.40:
        return 'Low resistant'
    else:
        return 'Unknown'

# Add the function of drug susceptibility classification to adata.obs
def add_sensitivity_classification(adata, sensitivity_col, drug):
    """
    Add the drug susceptibility classification to adata.obs.
    :param adata: AnnData object
    :param sensitivity_col: Drug sensitivity column name
    :param drug: Drug name
    :return: adata
    """
    sensitivity_class_col = f"{drug}_Sensitivity"  # Independent column name
    adata.obs[sensitivity_class_col] = adata.obs[sensitivity_col].apply(classify_sensitivity)
    return adata

# Calculate the function of spatial autocorrelation
def calculate_spatial_autocorrelation(adata, cell_type, sensitivity_col, drug, k=6):
    """
    Calculate the global spatial autocorrelation for a given cell type.
    :param adata: AnnData object
    :param cell_type: Cell type
    :param sensitivity_col: Drug sensitivity column name
    :param drug: Drug name
    :param k: Number of neighbors for KNN
    """
    sensitivity_class_col = f"{drug}_Sensitivity"  # Independent column name
    cell_type_indices = adata.obs[adata.obs[sensitivity_class_col] == cell_type].index
    cell_type_data = adata.obs.loc[cell_type_indices, sensitivity_col]

    if cell_type_data.empty:
        print(f"No data for {cell_type} in {drug}. Skipping spatial autocorrelation calculation.")
        return

    cell_type_indices = [adata.obs.index.get_loc(idx) for idx in cell_type_indices]
    cell_type_coords = adata.obsm['spatial'][cell_type_indices]

    # Check if coordinates are valid
    if np.isnan(cell_type_coords).any() or np.isinf(cell_type_coords).any():
        print(f"Invalid coordinates for {cell_type} in {drug}. Skipping spatial autocorrelation calculation.")
        return

    try:
        cell_type_weights = KNN.from_array(cell_type_coords, k=k)
    except Exception as e:
        print(f"Error calculating KNN for {cell_type} in {drug}: {e}")
        return

    moran = Moran(cell_type_data, cell_type_weights)
    print(f"Moran's I for {cell_type} in {drug}: {moran.I}, p-value: {moran.p_sim}")

# Organize the spatial autocorrelation results of each drug
def summarize_spatial_autocorrelation(adata, drugs, sensitivity_col_template="{}_Drug_Sensitivity_Predictions", k=6, save_folder=None):
    """
    Organize the spatial autocorrelation results of each drug and save to CSV.
    :param adata: AnnData object
    :param drugs: List of drugs
    :param sensitivity_col_template: Template for drug sensitivity column names
    :param k: Number of neighbors for KNN
    :param save_folder: Folder path to save CSV files
    :return: DataFrame of spatial autocorrelation results
    """
    results = []

    for drug in drugs:
        sensitivity_col = sensitivity_col_template.format(drug)
        sensitivity_class_col = f"{drug}_Sensitivity"  # Independent column name
        print(f"Processing drug: {drug}")

        # Add sensitivity classifications to an independent column
        adata = add_sensitivity_classification(adata, sensitivity_col, drug)

        # Spatial autocorrelation results
        drug_results = {
            "Drug": drug,
            "Global": [],
            "Local": [],
            "Tumor_High_Sensitive": [],
            "Region_CellType": []
        }

        # Calculate global and local autocorrelations
        cell_types = ['High sensitive', 'Low sensitive', 'Uncertain', 'High resistant', 'Low resistant', 'Unknown']
        for cell_type in cell_types:
            cell_type_indices = adata.obs[adata.obs[sensitivity_class_col] == cell_type].index
            cell_type_data = adata.obs.loc[cell_type_indices, sensitivity_col]

            if cell_type_data.empty:
                print(f"No data for {cell_type} in {drug}. Skipping spatial autocorrelation calculation.")
                drug_results["Global"].append({"CellType": cell_type, "Moran_I": np.nan, "p-value": np.nan})
                continue

            cell_type_indices = [adata.obs.index.get_loc(idx) for idx in cell_type_indices]
            cell_type_coords = adata.obsm['spatial'][cell_type_indices]

            # Check if coordinates are valid
            if np.isnan(cell_type_coords).any() or np.isinf(cell_type_coords).any():
                print(f"Invalid coordinates for {cell_type} in {drug}. Skipping spatial autocorrelation calculation.")
                drug_results["Global"].append({"CellType": cell_type, "Moran_I": np.nan, "p-value": np.nan})
                continue

            try:
                cell_type_weights = KNN.from_array(cell_type_coords, k=k)
            except Exception as e:
                print(f"Error calculating KNN for {cell_type} in {drug}: {e}")
                drug_results["Global"].append({"CellType": cell_type, "Moran_I": np.nan, "p-value": np.nan})
                continue

            # Global Moran's I
            moran = Moran(cell_type_data, cell_type_weights)
            drug_results["Global"].append({"CellType": cell_type, "Moran_I": moran.I, "p-value": moran.p_sim})

        results.append(drug_results)

    # Convert results to DataFrame
    global_results = []
    for drug_result in results:
        for global_result in drug_result["Global"]:
            global_results.append({
                "Drug": drug_result["Drug"],
                "CellType": global_result["CellType"],
                "Moran_I": global_result["Moran_I"],
                "p-value": global_result["p-value"]
            })

    global_df = pd.DataFrame(global_results)

    # Save to CSV if save_folder is provided
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        csv_path = os.path.join(save_folder, "SpatialAutocorrelation.csv")
        global_df.to_csv(csv_path, index=False)
        print(f"Spatial autocorrelation results saved to {csv_path}")

    return global_df

# Calculate tumor sensitivity means
def calculate_tumor_sensitivity_means(adata, drugs, sensitivity_col_template="{}_Drug_Sensitivity_Predictions", save_folder=None):
    """
    Calculate the mean sensitivity of each drug in the Tumor region.
    :param adata: AnnData object
    :param drugs: List of drugs
    :param sensitivity_col_template: Template for drug sensitivity column names
    :param save_folder: Folder path to save CSV files
    :return: tumor_sensitivity_means_df, top_2_drugs, bottom_2_drugs
    """
    if 'Region' not in adata.obs.columns:
        raise KeyError("'Region' column not found in adata.obs. Please ensure the 'Region' column is correctly loaded.")

    # Store mean sensitivity values for each drug in Tumor region
    tumor_sensitivity_means = []

    for drug in drugs:
        sensitivity_col = sensitivity_col_template.format(drug)
        
        # Check if drug sensitivity column exists
        if sensitivity_col not in adata.obs.columns:
            print(f"Column {sensitivity_col} not found in adata.obs. Skipping {drug}.")
            continue

        # Filter cells in Tumor region
        tumor_cells = adata.obs[adata.obs['Region'] == 'Tumor']
        if tumor_cells.empty:
            print(f"No Tumor region cells found for {drug}. Skipping.")
            continue

        # Calculate mean sensitivity in Tumor region
        tumor_sensitivity_mean = tumor_cells[sensitivity_col].mean()
        tumor_sensitivity_means.append({"Drug": drug, "Tumor_Mean_Sensitivity": tumor_sensitivity_mean})

    # Convert results to DataFrame
    tumor_sensitivity_means_df = pd.DataFrame(tumor_sensitivity_means)

    # Sort by mean sensitivity
    tumor_sensitivity_means_df = tumor_sensitivity_means_df.sort_values(by="Tumor_Mean_Sensitivity", ascending=False)

    # Get top 2 most sensitive drugs
    top_2_drugs = tumor_sensitivity_means_df.head(2)
    # Get bottom 2 least sensitive drugs
    bottom_2_drugs = tumor_sensitivity_means_df.tail(2)

    # Save as CSV files
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        # Save mean sensitivity for all drugs
        tumor_sensitivity_means_path = os.path.join(save_folder, "Tumor_Sensitivity_Means.csv")
        tumor_sensitivity_means_df.to_csv(tumor_sensitivity_means_path, index=False)
        print(f"Tumor sensitivity means saved to {tumor_sensitivity_means_path}")

        # Save top 2 most sensitive drugs
        top_2_drugs_path = os.path.join(save_folder, "Top_2_Sensitivity_Drugs.csv")
        top_2_drugs.to_csv(top_2_drugs_path, index=False)
        print(f"Top 2 sensitivity drugs saved to {top_2_drugs_path}")

        # Save bottom 2 least sensitive drugs
        bottom_2_drugs_path = os.path.join(save_folder, "Bottom_2_Sensitivity_Drugs.csv")
        bottom_2_drugs.to_csv(bottom_2_drugs_path, index=False)
        print(f"Bottom 2 sensitivity drugs saved to {bottom_2_drugs_path}")

    return tumor_sensitivity_means_df, top_2_drugs, bottom_2_drugs

# Calculate deconvolution-sensitivity correlation
def calculate_deconvolution_sensitivity_correlation(adata, deconv_columns, sensitivity_col_template="{}_Drug_Sensitivity_Predictions", save_folder=None):
    """
    Calculate correlation between deconvolution results and drug sensitivity predictions.
    :param adata: AnnData object
    :param deconv_columns: List of deconvolution result column names (e.g., ['Malignant', 'CAF', 'Endothelial', ...])
    :param sensitivity_col_template: Template for drug sensitivity column names
    :param save_folder: Folder path to save CSV files
    :return: DataFrame of correlation results
    """
    correlation_results = []

    # Iterate through all drugs
    for drug in drugs:
        sensitivity_col = sensitivity_col_template.format(drug)
        
        # Check if drug sensitivity column exists
        if sensitivity_col not in adata.obs.columns:
            print(f"Column {sensitivity_col} not found in adata.obs. Skipping {drug}.")
            continue

        # Iterate through all deconvolution columns
        for cell_type in deconv_columns:
            if cell_type not in adata.obs.columns:
                print(f"Column {cell_type} not found in adata.obs. Skipping {cell_type}.")
                continue

            # Calculate Pearson correlation coefficient
            correlation, p_value = stats.pearsonr(adata.obs[cell_type], adata.obs[sensitivity_col])

            # Store results
            correlation_results.append({
                "Drug": drug,
                "CellType": cell_type,
                "Correlation": correlation,
                "p-value": p_value
            })

    # Convert results to DataFrame
    correlation_df = pd.DataFrame(correlation_results)

    # Save as CSV file
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        csv_path = os.path.join(save_folder, "Deconvolution_Sensitivity_Correlation.csv")
        correlation_df.to_csv(csv_path, index=False)
        print(f"Deconvolution-Sensitivity correlation results saved to {csv_path}")

    return correlation_df

# Main function, run all steps
def main(file_paths, drugs):
    """
    The main function, which runs all the steps.
    :param file_paths: List of file paths [adata_path, deconv_path, interface_path]
    :param drugs: List of drugs
    :return: adata
    """
    adata_path, deconv_path, interface_path = file_paths

    # Load data
    adata, deconv_data, interface_data = load_files(adata_path, deconv_path, interface_path)

    # Handle deconvoluted data
    deconv_df = process_deconvolution_data(deconv_data, adata.obs.index)

    # Define the columns that need to be converted to floating-point numbers
    columns_to_convert = ['Malignant', 'CAF', 'Endothelial', 'Plasma', 'B cell', 'T CD4', 'T CD8', 'NK',
                          'cDC', 'pDC', 'Macrophage', 'Mast', 'Neutrophil']

    # Update OBS
    adata = update_obs_with_deconvolution(adata, deconv_df, columns_to_convert)

    for drug in drugs:
        sensitivity_col = f"{drug}_Drug_Sensitivity_Predictions"

        # Add sensitivity classifications
        adata = add_sensitivity_classification(adata, sensitivity_col, drug)

        # Calculate Moran's I spatial autocorrelation
        cell_types = ['High sensitive', 'Low sensitive', 'Uncertain', 'High resistant', 'Low resistant', 'Unknown']
        for cell_type in cell_types:
            calculate_spatial_autocorrelation(adata, cell_type, sensitivity_col, drug)

        # Make sure that the length of the interface_data is the same as that of adata.obs
        print(f"Length of adata.obs: {len(adata.obs)}")
        print(f"Length of interface_data: {len(interface_data)}")

        # If interface_data has a 'Cell_id' column, make sure it matches the adata.obs index
        if 'Cell_id' in interface_data.columns:
            interface_data = interface_data.set_index('Cell_id')
            interface_data = interface_data.reindex(adata.obs.index)

        # If interface_data has more rows than adata.obs, only keep rows matching adata.obs index
        if len(interface_data) > len(adata.obs):
            interface_data = interface_data.loc[adata.obs.index]

        # Load interface data and add to adata.obs['Region']
        adata.obs['Region'] = pd.Categorical(interface_data['InterfaceType'])

    return adata

# Plot function
def plot_drug_sensitivity(adata, drug, save_folder):
    """
    Plot drug sensitivity distributions and save the plot.
    :param adata: AnnData object
    :param drug: Drug name
    :param save_folder: Folder path to save images
    """
    sensitivity_col = f"{drug}_Drug_Sensitivity_Predictions"
    sc.pl.spatial(adata, color=sensitivity_col, show=False)
    
    # Save as PDF and TIFF
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, f"{drug}_sensitivity.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, f"{drug}_sensitivity.tiff"), format='tiff', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sensitivity_classification(adata, drug, palette, save_folder):
    """
    Plot sensitivity classification distribution for a specific drug and save the plot.
    :param adata: AnnData object
    :param drug: Drug name
    :param palette: Color palette
    :param save_folder: Folder path to save images
    """
    sensitivity_col = f"{drug}_Drug_Sensitivity_Predictions"
    sensitivity_class_col = f"{drug}_Sensitivity"  # Independent column name
    adata.obs[sensitivity_class_col] = adata.obs[sensitivity_col].apply(classify_sensitivity)
    sc.pl.spatial(adata, color=sensitivity_class_col, palette=palette, show=False)
    
    # Save as PDF and TIFF
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, f"{drug}_sensitivity_classification.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, f"{drug}_sensitivity_classification.tiff"), format='tiff', dpi=300, bbox_inches='tight')
    plt.close()

def plot_region_distribution(adata, palette, save_folder):
    """
    Plot region distribution and save the plot.
    :param adata: AnnData object
    :param palette: Color palette
    :param save_folder: Folder path to save images
    """
    sc.pl.spatial(adata, color='Region', palette=palette, show=False)
    
    # Save as PDF and TIFF
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, "Region_distribution.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, "Region_distribution.tiff"), format='tiff', dpi=300, bbox_inches='tight')
    plt.close()

