Spatially resolved prediction of drug sensitivity in the tumor microenvironment via deep graph contrastive and transfer learning
-
Background
-
Tumor microenvironment heterogeneity remains a key barrier to precision oncology. While single-cell RNA sequencing (scRNA-seq) captures intratumoral gene expression diversity, existing drug sensitivity models often overlook spatial variation in drug-responsive cell populations across tumor–stroma immune interface. As a result, they may overestimate overall sensitivity despite low local response or weak spatial clustering, compromising therapeutic efficacy.

Methods
-
To address this challenge, we present DeepTNR (Deep Learning for Tumor Niche Response), an innovative computational framework that integrates deep graph contrastive learning with transfer learning to model tumor-microenvironment interactions. The predicted drug sensitivity profiles are then combined with tumor-stroma immune interface features and spatial autocorrelation analysis to identify the most effective therapeutic candidates. We applied DeepTNR to spatial transcriptomic datasets from nine tumor types and selected colorectal cancer (CRC) as a case study due to its pronounced spatial heterogeneity and microinfiltrative architecture. 

Results
-
Analysis of eight colorectal cancer (CRC) specimens revealed distinct spatial patterns of drug sensitivity, influenced by the compartmental organization of malignant, stromal, and immune components. Carmustine exhibited significant sensitivity across the tumor region, with sensitive cells also detected in focal stromal microinfiltration zones, suggesting its potential to target both the tumor core and invasive margins. In contrast, docetaxel-sensitive cells were broadly distributed within tumor regions but largely absent from stromal microinfiltration zones, indicating limited efficacy against invasive tumor fronts that may harbor resistant subpopulations. Spatial autocorrelation analysis further supported these findings, with carmustine-sensitive cells showing strong clustering within tumor regions, consistent with its localized efficacy. However, high spatial autocorrelation alone can be misleading. In the case of axitinib, sensitive cells also exhibited strong clustering and high abundance, yet were predominantly localized within the stroma rather than the tumor core, highlighting the need to interpret spatial patterns in the context of tumor architecture. These findings underscore the importance of spatial autocorrelation as a quantitative metric to capture spatially confined drug sensitivity and to distinguish between locally concentrated and peripherally scattered therapeutic responses. 

Conclusion
-
This method captures both gene expression profiles and spatial distribution patterns within tissues, advancing the effectiveness of personalized cancer therapies.
![Fig 1](https://github.com/user-attachments/assets/414940e5-8642-4730-aabf-61aba273ff90)

Step 0 "Installation and setup"
-     
  1.Set up the Python environment
-
```shell
git clone https://github.com/ricardoliuhong/DeepTNR.git
conda env create -f environment.yaml
conda activate DeepTNR
```
  2.Set up the R environment
 -
```R
#R version = 4.40
# install.packages("devtools")
remotes::install_version("SeuratObject", "4.1.4", repos = c("https://satijalab.r-universe.dev", getOption("repos")))
remotes::install_version("Seurat", "4.4.0", repos = c("https://satijalab.r-universe.dev", getOption("repos")))
devtools::install_github("data2intelligence/SpaCET")
if (!require("remotes")) install.packages("remotes")
versions <- list(
  cowplot = "1.1.1",
  dplyr = "1.1.3",
  hdf5r = "1.3.11"
)
lapply(names(versions), function(pkg) {
  remotes::install_version(
    pkg,
    version = versions[[pkg]],
    dependencies = TRUE,
    upgrade = "never"
  )
})

```

  3.Download sample data
-
Example data including spatial transcriptome profiles and cell line RNA expression data can be accessed at: [Data](https://drive.google.com/drive/folders/1h1RgI21EHF5ndKqlnwvj5-itj1cWAo11?usp=sharing) Then put the data files into the “Spatial Transcriptomics Data Preprocessing” folder.



Step 1 "Spatial Transcriptomics Data Preprocessing"
-     
  1.Initial data processing step in R
-
```r

source("ST_interface.R")
source("ST_pre.R")
data_dirs <- list.dirs(path = ".", full.names = FALSE, recursive = FALSE)
data_dirs <- data_dirs[grepl("^VISDS", data_dirs)] ######          Filter folders starting with "VISDS" dataset
for (dir in data_dirs) {
  process_data_set(dir)
}
```

 2.Get tumor-stroma immune interface in Python
-
```python
import Interface as is
is.interface('CRC2.h5ad', 'VISDS000772_interface_data.csv', 'CRC2_annotated.h5ad')

```

Step 2 Predicting drug sensitivity in spatial transcriptomics of tumors via deep graph contrastive and transfer learning"
-   

  1.Create feature graphs in Python   
-

```shell

drugs=("X5.FLUOROURACIL" "AZ960" "AZD2014" "AZD4547" "AZD5363" "AZD5438" "AZD6482" "AZD7762" "AZD8055" "BI.2536" "BIBR.1532" "BMS.345541" "BMS.754807" "GSK1904529A" "I.BET.762" "LY2109761" "MG.132" "MK.1775" "MK.2206" "OSI.027" "OTX015" "P22077" "PLX.4720" "PRT062607" "VE.822" "AFATINIB" "ALISERTIB" "ALPELISIB" "AXITINIB" "BORTEZOMIB" "BUPARLISIB" "CAMPTOTHECIN" "CARMUSTINE" "CEDIRANIB" "CISPLATIN" "CRIZOTINIB" "CYCLOPHOSPHAMIDE" "CYTARABINE" "DABRAFENIB" "DASATINIB" "DINACICLIB" "DOCETAXEL" "ENTINOSTAT" "EPIRUBICIN" "ERLOTINIB" "FLUDARABINE" "FORETINIB" "FULVESTRANT" "GEFITINIB" "GEMCITABINE" "IBRUTINIB" "IRINOTECAN" "LAPATINIB" "LINSITINIB" "MITOXANTRONE" "NAVITOCLAX" "NELARABINE" "NILOTINIB" "NIRAPARIB" "OLAPARIB" "OSIMERTINIB" "OXALIPLATIN" "PACLITAXEL" "PALBOCICLIB" "PEVONEDISTAT" "RIBOCICLIB" "RUXOLITINIB" "SELUMETINIB" "SORAFENIB" "TALAZOPARIB" "TAMOXIFEN" "TASELISIB" "TEMOZOLOMIDE" "TENIPOSIDE" "TOPOTECAN" "TOZASERTIB" "TRAMETINIB" "UPROSERTIB" "VENETOCLAX" "VINBLASTINE" "VINCRISTINE" "VINORELBINE" "VORINOSTAT")

expid=1
device='cuda:0'
val_dataset='CRC2_val_adata.h5ad'  ### Modify the prefix when the data changes.  
train_dataset='Bulk_train_adata.h5ad' ###No changes needed
train_expression_file='Bulk_features.csv' ###No changes needed
train_binary_labels_file='Bulk_train_labels.csv' ###No changes needed
val_binary_labels_file='Bulk_val_labels.csv' ###No changes needed
Spatial_dataset='CRC2.h5ad' ### Modify the prefix when the data changes.
expression_file='ALL_expression.csv'  ###No changes needed
binary_labels_file='ALL_label_binary_wf.csv'
output_prefix='CRC2'  ### sample name
visium_path='VISDS000772' ###  ST access number
count_file='filtered_feature_bc_matrix.h5' ###No changes needed
output_file='CRC2_DeepTNR.h5ad'     ###Modify the prefix when the data changes.

for drug in "${drugs[@]}"
do
    echo "Processing drug: $drug"
    
    # Run Python script
    python -u DeepTNR_pre.py \
        --expid $expid \
        --device $device \
        --train_dataset $train_dataset \
        --train_expression_file $train_expression_file \
        --train_binary_labels_file $train_binary_labels_file \
        --Spatial_dataset $Spatial_dataset \
        --expression_file $expression_file \
        --output_prefix $output_prefix \
        --Drug $drug \
        --visium_path $visium_path \
        --count_file $count_file \
        --output_file $output_file
    
    echo "Finished processing drug: $drug"
done
```




  2.Predicting drug sensitivity 
 - 
In python
```shell
device='cuda:0'
Spatial_dataset='CRC2_DeepTNR.h5ad'
for drug in "${drugs[@]}"
do
    echo "Running DeepTNR.py for drug: $drug"
    python -u DeepTNR.py \
        --source_features "DeepTNR_Data/${drug}_Bulk_features.npy" \
        --target_features "DeepTNR_Data/scRNA_features.npy" \
        --source_edge_index "DeepTNR_Data/${drug}_Bulk_edge_index.npy" \
        --target_edge_index "DeepTNR_Data/scRNA_edge_index.npy" \
        --Spatial_dataset "DeepTNR_Data/$Spatial_dataset" \
        --Drug $drug \
        --source_labels "DeepTNR_Data/${drug}_Bulk_labels.npy" \
        --model "GAT" \ 
        --device $device  

    echo "Finished running crossgraph.py for drug: $drug"
done
```
  3.Visualization of predicted results
 - 
 In python
 ###Here we use the CEDIRANIB prediction results as an example
```Python
import Visualization as vis
file_paths = [
    "CRC2_DeepTNR.h5ad",      # Replace with your actual adata file path
    "VISDS000772_prop_mat.csv", # Replace with your deconvolution file path
    "VISDS000772_interface_data.csv"      # Replace with your interface file path
]
drugs = ["CRIZOTINIB"]
cancer_sample_name = "CRC2"


file_paths = tuple(file_paths)
Results_folder = f"{cancer_sample_name}_Result"
SpatialAutocorrelation = f"{Results_folder}/{cancer_sample_name}_SpatialAutocorrelation"
# Run the main function
adata = vis.main(file_paths, drugs)
# Organize spatial autocorrelation results and save to CSV
summary_df = vis.summarize_spatial_autocorrelation(adata, drugs, save_folder=SpatialAutocorrelation)
# Print the finishing result
print(summary_df)
# Define color palettes
Drug_Sensitivity_map = {
    '1.0': '#E69f00',  # Orange
    '0.0': '#56b4e9'   # Blue
}
Cell_color = {
    'High sensitive': '#FF0000',
    'Low sensitive': '#e85a71',
    'Uncertain': '#D3D3D3',
    'High resistant': '#0072B2',
    'Low resistant': '#87CEEB',
    'Unknown': '#808080'
}
Region_color = {
    'Stroma': '#56b4e9',
    'Tumor': '#E69f00',
    'Interface': '#EE0C0C'
}
# Plot drug sensitivity distributions
for drug in drugs:
    vis.plot_drug_sensitivity(adata, drug, Results_folder)
# Plot the distribution of sensitivity classifications
for drug in drugs:
    vis.plot_sensitivity_classification(adata, drug, Cell_color, Results_folder)
# Plot the regional distribution
vis.plot_region_distribution(adata, Region_color, Results_folder)
# Calculate tumor sensitivity means and export the top 2 and bottom 2 drugs
tumor_sensitivity_means_df, top_2_drugs, bottom_2_drugs = vis.calculate_tumor_sensitivity_means(adata, drugs, save_folder=Results_folder)
# Print the results
print("\nTop 2 drugs with the highest mean sensitivity in Tumor region:")
print(top_2_drugs)
print("\nBottom 2 drugs with the lowest mean sensitivity in Tumor region:")
print(bottom_2_drugs)
```
![CEDIRANIB_sensitivity](https://github.com/user-attachments/assets/fb46c2f7-41a5-4755-bfea-54eb38de5c90)
![CRC2_CEDIRANIB_sensitivity_classification](https://github.com/user-attachments/assets/1e85e928-811d-4da0-a5ee-f51b2578a5ce)
  
  4.Infer spatial autocorrelation
-
In python  
```python
import Spatial_autocorrelation as sa
CRC2_csv_path = "CRC2_SpatialAutocorrelation.csv"  
CRC2_folder = "CRC2_Result"  
sa.plot_spatial_autocorrelation_for_drug(CRC2_csv_path, "CEDIRANIB", CRC2_folder)
```
![CEDIRANIB_Spatial_Autocorrelation](https://github.com/user-attachments/assets/cd0fde7f-bf58-403b-9436-095bd3aa5703)

Step 3 "Downstream analyses "  We have built upon existing advanced spatial transcriptomics tools as a complement to DeepTNR to investigate the relationship between the tumor-stroma-immune interface and drug sensitivity. 
-
  1.Inferring the correlation between cell deconvolution results and sensitivity
-
In R
```R
library("SpaCET")
library("Seurat")
library("ggplot2")
library("cowplot")
library("dplyr")
library("hdf5r")
#'Malignant', 'CAF', 'Endothelial', 'Plasma', 'B cell', 'T CD4', 'T CD8', 'NK', 
#'cDC', 'pDC', 'Macrophage', 'Mast', 'Neutrophil'
#'
VISDS000772_SpaCET_obj=readRDS("VISDS000772_SpaCET_obj.rds")
CRC2=VISDS000772_SpaCET_obj
pdf("Malignant.pdf", width = 10, height = 8)  
SpaCET.visualize.spatialFeature(
  CRC2, 
  spatialType = "CellFraction", 
  spatialFeatures=c("Malignant")
)
```
  2.Inferring cancer cell status and its correlation with sensitivity   
 - 
In R
 
  ## These are the tumor cell states:   "Cycle, Stress, Interferon, Hypoxia, Oxphos, Metal, cEMT, pEMT, Alveolar, Basal, Squamous, Glandular, Ciliated, AC, OPC, NPC"
  

```R

library(SpaCET)
library(Seurat)
library(dplyr)
VISDS000772_SpaCET_obj=readRDS("VISDS000772_SpaCET_obj.rds")
CRC2=VISDS000772_SpaCET_obj
# run gene set calculation
gmt2 <- read.gmt("CancerCellState.gmt")
CRC2 <- SpaCET.GeneSetScore(CRC2, GeneSets = gmt2)
CRC2@results$GeneSetScore[1:6,1:6]
rownames(CRC2@results$GeneSetScore)
output_dir <- "CRC2_GeneSet_Plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}
# Cycle
pdf(file = file.path(output_dir, "CRC2_Cycle.pdf"), width = 6.2, height = 5)
SpaCET.visualize.spatialFeature(
  CRC2, 
  spatialType = "GeneSetScore",
  spatialFeatures = "Cycle"
)
dev.off()
```
  3.Inferring the direction and strength of spatial transcriptome signaling flows
-
In python
```
import os
import gc
import ot
import pickle
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

import commot as ct
adata=sc.read("CRC2.h5ad")

adata.raw = adata
adata_dis500 = adata.copy()

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.4)


df_cellchat = ct.pp.ligand_receptor_database(species='human', signaling_type='Secreted Signaling', database='CellChat')
print(df_cellchat.shape)
df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, adata_dis500, min_cell_pct=0.05)
print(df_cellchat_filtered.shape)
print(df_cellchat_filtered.head())
df_cellchat.to_csv('CRC2_df_cellchat.csv', index=False)
ct.tl.spatial_communication(
    adata_dis500,
    database_name='cellchat',
    df_ligrec=df_cellchat_filtered,
    dis_thr=500,
    heteromeric=True,
    pathway_sum=True
)

ct.tl.communication_direction(
    adata_dis500,
    database_name='cellchat',
    pathway_name='MK', 
    k=5
)
ct.pl.plot_cell_communication(
    adata_dis500,
    database_name='cellchat',
    pathway_name='MK',  
    plot_method='grid',
    background_legend=True,
    scale=0.0003,
    ndsize=4,
    grid_density=0.4,
    summary='sender',
    background='image',
    clustering='leiden',
    cmap='Alphabet',
    normalize_v=True,
    normalize_v_quantile=0.995,
    filename='CRC2_MK.pdf'
)
``` 
     
    
  

