rm(list=ls())
library(FactoMineR)

#Read the data
AllWineData <- read.table("AllWineDataPreProcessed.csv", header=TRUE, sep=";")
indnames <- rownames(AllWineData)
varnames <- colnames(AllWineData)
  
#PCA
pca <-PCA(AllWineData, quali.sup = 13, quanti.sup = 12, scale = T) #Shouldn't type and quality be supplementary?

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