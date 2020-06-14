rm(list=ls())
library(FactoMineR)
library(factoextra)

#Read the data
AllWineData <- read.table("AllWineDataPreProcessed.csv", header=TRUE, sep=";")
indnames <- rownames(AllWineData)
varnames <- colnames(AllWineData)

#PCA
pca <-PCA(AllWineData, quali.sup = 13, quanti.sup = 12, scale = T, graph = F) #Shouldn't type and quality be supplementary?

AllWineData$taste <- ifelse(AllWineData$quality < 5, "bad", "good")
AllWineData$taste[AllWineData$quality == 5] <- "normal"
AllWineData$taste[AllWineData$quality == 6] <- "normal"
AllWineData$taste[AllWineData$quality >= 8] <- "excellent"
AllWineData$taste <- as.factor(AllWineData$taste)

#Visualization
fviz_pca_ind(pca, geom = "point", col.ind = AllWineData$taste)

#HCPC clustering
AllWineData.hcpc <- HCPC(pca, nb.clust=-1, consol=T)

#paragons
AllWineData.hcpc$desc.ind$para #Useless

#Profiling
cut <- AllWineData.hcpc$data.clust$clust
catdes <- catdes(cbind(as.factor(cut),AllWineData),1, proba = 0.0005)
#
catdes$category #red or white
catdes$quanti #rest of influenciable variables