
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text



def interface(h5ad_path, csv_path, output_h5ad_path):

    adata = sc.read_h5ad(h5ad_path)
    
    # Read the CSV file
    interface_data = pd.read_csv(csv_path)
    
    # Ensure data types are consistent
    interface_data['X'] = interface_data['X'].astype(int)
    interface_data['Y'] = interface_data['Y'].astype(int)
    adata.obs['array_row'] = adata.obs['array_row'].astype(int)
    adata.obs['array_col'] = adata.obs['array_col'].astype(int)
    
    # Merge InterfaceType annotation into adata.obs
    merged_data = adata.obs.merge(interface_data[['X', 'Y', 'InterfaceType']], 
                                  left_on=['array_row', 'array_col'], 
                                  right_on=['X', 'Y'], 
                                  how='left')
    
    # Drop rows where InterfaceType is NA
    filtered_data = merged_data.dropna(subset=['InterfaceType'])
    
    # Drop unnecessary X and Y columns
    filtered_data.drop(columns=['X', 'Y'], inplace=True)
    
    # Update adata.obs with the filtered data
    adata.obs = filtered_data
    
    # Save the annotated AnnData object

    adata.write_h5ad(output_h5ad_path)
