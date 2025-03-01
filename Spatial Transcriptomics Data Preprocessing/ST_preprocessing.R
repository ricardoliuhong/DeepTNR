library(SpaCET)
library(Seurat)
library(dplyr)

# 定义一个函数来处理数据集
process_dataset <- function(data_dir, visiumPath, slice_name) {
  # 加载10X空间数据
  CRC <- Load10X_Spatial(data.dir = data_dir,
                         filename = "filtered_feature_bc_matrix.h5",
                         assay = "spatial", 
                         slice = slice_name)
  
  # 获取之前保存的SpaCET对象
  SpaCET_obj <- readRDS(paste0(visiumPath, "_SpaCET_obj.rds"))
  
  # 将SpaCET对象的数据添加到Seurat对象
  CRC <- addTo.Seurat(SpaCET_obj, CRC)
  
  # 设置默认Assay
  Seurat::DefaultAssay(CRC) <- "propMatFromSpaCET"
  
  # 提取细胞比例矩阵
  prop_mat <- CRC@assays[["propMatFromSpaCET"]]@data
  
  # 获取接口值
  interface_values <- SpaCET_obj@results$CCI$interface[1, ]
  
  # 获取空间点的坐标
  spots <- colnames(SpaCET_obj@results$CCI$interface)
  
  # 将空间点的坐标字符串分割为X和Y坐标
  coordinates <- strsplit(spots, "x")
  x_coords <- as.numeric(sapply(coordinates, `[`, 1))
  y_coords <- as.numeric(sapply(coordinates, `[`, 2))
  
  # 创建数据框
  df <- data.frame(
    Spot = spots,
    X = x_coords,
    Y = y_coords,
    InterfaceType = interface_values
  )
  
  # 输出数据框结构
  str(df)
  
  # 保存数据框到CSV文件
  write.csv(df, paste0(visiumPath, "_interface_data.csv"), row.names = FALSE)
  
  print(paste("Processed", visiumPath))
}

# 你的数据集文件夹路径
data_dir <- getwd()
# 数据集名称和对应的slice名称
datasets <- list(
  #"VISDS000771" = "CRC1",
  "VISDS000772" = "CRC2",
  "VISDS000773" = "CRC3",
  "VISDS000774" = "CRC4",
  "VISDS000775" = "CRC5",
  "VISDS000776" = "CRC6",
  "VISDS000777" = "CRC7"
  #"VISDS000778" = "CRC8"
  # 在这里可以添加更多的数据集和对应的slice名称
)

# 循环处理每个数据集
for (visiumPath in names(datasets)) {
  process_dataset(data_dir, visiumPath, datasets[[visiumPath]])
}

print("All datasets processed successfully!")
