The complexity of tumor heterogeneity presents a significant challenge to precision oncology. Single-cell RNA sequencing (scRNA-seq) has achieved remarkable advances in uncovering subtle gene expression variations within and beyond tumors, driving personalized cancer treatment. Spatial transcriptomics (ST) complements this by precisely mapping gene expression in intact tissues, deepening our understanding of tumor heterogeneity. ST enables researchers to identify malignant features at single-cell or regional levels, providing essential insights for developing targeted therapies. Despite the availability of tools to predict drug sensitivity at the single-cell level, they often overlook spatial differences in the distribution of drug-sensitive and drug-resistant cells. This can lead to predictions of high overall sensitivity, while specific tumor regions exhibit low sensitivity or weak spatial autocorrelation among sensitive cells, undermining treatment efficacy. To address this challenge, we introduce DeepTNR (Deep Learning for Tumor Niche Response), a novel computational framework that leverages deep graph contrastive learning and transfer learning. This method captures both gene expression profiles and spatial distribution patterns within tissues, advancing the effectiveness of personalized cancer therapies. We applied DeepTNR to the spatial transcriptome profiles of nine cancer types, utilizing a dataset comprising 37 samples and 81,667 cell spots, with colorectal cancer (CRC) highlighted as a case study.
![Fig 1](https://github.com/user-attachments/assets/414940e5-8642-4730-aabf-61aba273ff90)

Step1 "Spatial Transcriptomics Data Preprocessing"
=     
  1.Initial data processing step in R
-
```r

source("ST_interface.R")
source("ST_pre.R")
data_dirs <- list.dirs(path = ".", full.names = FALSE, recursive = FALSE)
data_dirs <- data_dirs[grepl("^VISDS", data_dirs)] ######       Filter folders starting with "VISDS" dataset
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


 
     
    
  

