import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define color mapping
sensitivity_colors = {
    'High sensitive': '#FF0000',  # Red
    'Low sensitive': '#e85a71',   # Light Red
    'Uncertain': '#D3D3D3',       # Gray
    'High resistant': '#0072B2',  # Dark Blue
    'Low resistant': '#87CEEB',   # Light Blue
}

def plot_spatial_autocorrelation_for_drug(csv_path, drug_name, save_folder=None):
    """
    Read data from a saved CSV file and plot the spatial autocorrelation for the specified drug.
    :param csv_path: Path to the CSV file containing spatial autocorrelation results
    :param drug_name: Name of the specified drug
    :param save_folder: Folder path to save the output image
    """
    # Read the CSV file
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} does not exist. Please run spatial autocorrelation analysis first.")

    global_df = pd.read_csv(csv_path)

    # Filter data for the specified drug
    drug_df = global_df[global_df["Drug"] == drug_name]
    if drug_df.empty:
        print(f"No data found for drug {drug_name}.")
        return

    # Remove "Unknown" categories and invalid cell types
    drug_df = drug_df[drug_df["CellType"] != "Unknown"]
    drug_df = drug_df.dropna(subset=["CellType", "Moran_I"])  # Remove records with NaN values in CellType or Moran_I

    # If no valid data remains, exit the function
    if drug_df.empty:
        print(f"No valid cell type data for drug {drug_name}.")
        return

    # Sort by CellType and Moran's I
    drug_df = drug_df.sort_values(by=["CellType", "Moran_I"], ascending=[True, False])

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,  # Global font size
        'axes.titlesize': 16,  # Title font size
        'axes.labelsize': 14,  # Axis label font size
        'xtick.labelsize': 12,  # X-axis tick label font size
        'ytick.labelsize': 12,  # Y-axis tick label font size
        'legend.fontsize': 12,  # Legend font size
    })

    # Create bar plot
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=drug_df,
        x="Moran_I",
        y="CellType",
        hue="CellType",
        orient="h",
        palette=sensitivity_colors,
        dodge=False  # Avoid grouped bar chart
    )
    
    plt.title(f"Global Moran's I for {drug_name}", fontsize=16)  # Title font size
    plt.xlabel("Moran's I", fontsize=14)  # X-axis label font size
    plt.ylabel("Cell Type", fontsize=14)  # Y-axis label font size
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save or display the chart
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{drug_name}_Spatial_Autocorrelation.png")
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Spatial autocorrelation plot saved to {save_path}")
    else:
        plt.show()
