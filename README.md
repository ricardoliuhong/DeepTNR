Predicting drug sensitivity with spatial precision across and beyond the tumor-stroma immune interface via deep graph contrastive and transfer learning
-
Abstract
-
The complexity of tumor heterogeneity presents a significant challenge to precision oncology. Single-cell RNA sequencing (scRNA-seq) has achieved remarkable advances in uncovering subtle gene expression variations within and beyond tumors, driving personalized cancer treatment. Spatial transcriptomics (ST) complements this by precisely mapping gene expression in intact tissues, deepening our understanding of tumor heterogeneity. ST enables researchers to identify malignant features at single-cell or regional levels, providing essential insights for developing targeted therapies. Despite the availability of tools to predict drug sensitivity at the single-cell level, they often overlook spatial differences in the distribution of drug-sensitive and drug-resistant cells. This can lead to predictions of high overall sensitivity, while specific tumor regions exhibit low sensitivity or weak spatial autocorrelation among sensitive cells, undermining treatment efficacy. To address this challenge, we introduce DeepTNR (Deep Learning for Tumor Niche Response), a novel computational framework that leverages deep graph contrastive learning and transfer learning. This method captures both gene expression profiles and spatial distribution patterns within tissues, advancing the effectiveness of personalized cancer therapies. We applied DeepTNR to the spatial transcriptome profiles of nine cancer types, utilizing a dataset comprising 37 samples and 81,667 cell spots, with colorectal cancer (CRC) highlighted as a case study.

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
Example data including spatial transcriptome profiles and cell line RNA expression data can be accessed at: [Data](https://drive.google.com/drive/folders/1h1RgI21EHF5ndKqlnwvj5-itj1cWAo11?usp=sharing)




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
import Interface
interface('CRC2.h5ad', 'VISDS000772_interface_data.csv', 'CRC2_annotated.h5ad')

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
    python -u pre.py \
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

```shell
device='cuda:0'
Spatial_dataset='CRC2_DeepTNR.h5ad'
for drug in "${drugs[@]}"
do
    echo "Running crossgraph.py for drug: $drug"
    python -u crossgraph.py \
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
 ###Here we use the CEDIRANIB prediction results as an example
```Python

```
![CRC2_CEDIRANIB_sensitivity_classification](https://github.com/user-attachments/assets/1e85e928-811d-4da0-a5ee-f51b2578a5ce)

Step 3 "Downstream analyses " 
-
  1.Spatial_Autocorrelation
 ```pythom 
CRC2_csv_path = "CRC2_SpatialAutocorrelation.csv"  
CRC2_folder = "CRC2_Result"  
plot_spatial_autocorrelation_for_drug(CRC2_csv_path, "CEDIRANIB", CRC2_folder)
```
![CEDIRANIB_Spatial_Autocorrelation](https://github.com/user-attachments/assets/cd0fde7f-bf58-403b-9436-095bd3aa5703)





 
     
    
  

