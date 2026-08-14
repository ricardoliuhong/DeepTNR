DeepTNR: predicting tumor drug sensitivity from spatial transcriptomics with deep graph contrastive and transfer learning
--
Step 0 "Installation and setup"
-     
  1.Set up the Python environment
-
```shell
git clone https://github.com/ricardoliuhong/DeepTNR.git
conda create -n deeptnr python=3.9 -y
conda activate DeepTNR
pip install -r requirements.txt

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

```python
import Interface as is
is.interface('CRC1.h5ad', 'VISDS000771_interface_data.csv', 'CRC1_annotated.h5ad')

```
 
 3.The tumor-stroma immune interface as revised based on pathologist's reference  : [Data][CRC1_region_interface.csv](https://github.com/user-attachments/files/27985160/CRC1_region_interface.csv)

 
 4.Visium HD tumor-periphery domain assignment in Python (Visium HD only)

For Visium HD 8 μm bins, `VisiumHD_domain.py` assigns three domain labels: Tumor, 50 micron, and Tissue. The implementation follows Oliveira et al. (Nat Genet 2025, Fig. 4). Place the Visium HD AnnData, optional deconvolution metadata parquet, and Space Ranger spatial files under your working directory, or pass absolute paths.


Example Visium HD inputs: [CRC1.h5ad](https://drive.google.com/drive/folders/1h1RgI21EHF5ndKqlnwvj5-itj1cWAo11?usp=drive_link); [P1CRC_Metadata.parquet](https://github.com/10XGenomics/HumanColonCancer_VisiumHD/raw/main/MetaData/P1CRC_Metadata.parquet); [tissue_positions.parquet.gz](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594567/suppl/GSM8594567_P1CRC_tissue_positions.parquet.gz); [scalefactors_json.json.gz](https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/GSM8594567/suppl/GSM8594567_P1CRC_scalefactors_json.json.gz). Put `tissue_positions.parquet` and `scalefactors_json.json` into `HumanColonCancer_VisiumHD/binned_outputs/square_008um/spatial/`.

```python
import VisiumHD_domain as vhd

vhd.visiumhd_domain(
    'CRC1.h5ad',
    metadata_path='P1CRC_Metadata.parquet',
    spaceranger_root='HumanColonCancer_VisiumHD',
    output_h5ad_path='CRC1_VisiumHD_domain.h5ad',
    distance_um=50,
    min_tumor_neighbors=25,
    # tumor_label='Tumor II',  # optional; leave unset to auto-detect from DeconvolutionLabel1
)
```

Or from the command line:

```shell
python -u VisiumHD_domain.py \
    --adata CRC1.h5ad \
    --metadata P1CRC_Metadata.parquet \
    --spaceranger-root HumanColonCancer_VisiumHD \
    --output-h5ad CRC1_VisiumHD_domain.h5ad \
    --outdir VisiumHD_domain_outputs \
    --distance-um 50 \
    --min-tumor-neighbors 25
```

The annotated AnnData stores domain labels in `adata.obs['VisiumHD_domain_py']`. Use this file for downstream drug-sensitivity comparison by region (Step 3).

Step 2 Predicting drug sensitivity in spatial transcriptomics of tumors via deep graph contrastive and transfer learning"
-   

  1.Create feature graphs in Python   


```shell

drugs=("X5.FLUOROURACIL" "IRINOTECAN" "OXALIPLATIN" )

expid=1
device='cuda:0'
val_dataset='CRC1_val_adata.h5ad'  ### Modify the prefix when the data changes.  
train_dataset='Bulk_train_adata.h5ad' ###No changes needed
train_expression_file='Bulk_features.csv' ###No changes needed
train_binary_labels_file='Bulk_train_labels.csv' ###No changes needed
val_binary_labels_file='Bulk_val_labels.csv' ###No changes needed
Spatial_dataset='CRC1.h5ad' ### Modify the prefix when the data changes.
expression_file='ALL_expression.csv'  ###No changes needed
binary_labels_file='ALL_label_binary_wf.csv'###No changes needed
output_prefix='CRC1'  ### sample name
visium_path='VISDS000771' ###  ST access number
count_file='filtered_feature_bc_matrix.h5' ###No changes needed
output_file='CRC1_DeepTNR.h5ad'     ###Modify the prefix when the data changes.

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
 
In python
```shell

drugs=("X5.FLUOROURACIL" "IRINOTECAN" "OXALIPLATIN" )
device='cuda:0'
scRNA_dataset='CRC1.h5ad'  

for drug in "${drugs[@]}"
do
    echo "Running DeepTNR.py for drug: $drug"
    python -u crossgraph.py \
        --source_features "DeepTNR_Data/${drug}_Bulk_features.npy" \
        --target_features "DeepTNR_Data/scRNA_features.npy" \
        --source_edge_index "DeepTNR_Data/${drug}_Bulk_edge_index.npy" \
        --target_edge_index "DeepTNR_Data/scRNA_edge_index.npy" \
        --scRNA_dataset "DeepTNR_Data/$scRNA_dataset" \
        --Drug $drug \
        --source_labels "DeepTNR_Data/${drug}_Bulk_labels.npy" \
        --model "GAT" \ 
        --device $device  

    echo "Finished running crossgraph.py for drug: $drug"
done






```
Step 3 "Visualize the prediction results and explore spatial autocorrelation and proximity effects!! 
-
Please view the DeepTNR_Tutorial.ipynb at: 
🔗 https://github.com/ricardoliuhong/DeepTNR/blob/8eec72c8889ecda43c0bd68c8cc5adbcb94ccd88/DeepTNR_Tutorial.ipynb
-
 And download the necessary data from: 
🔗 https://drive.google.com/drive/folders/1h1RgI21EHF5ndKqlnwvj5-itj1cWAo11?usp=drive_link
-
Happy analyzing! 🚀
-

## 📋 Tutorial Contents:
   1. 🎨 Visualization of Predicted Results
   2. 📍 Performing Spatial Autocorrelation Analysis
   3. 🔬 Performing Spatial Proximity Effect Analysis
