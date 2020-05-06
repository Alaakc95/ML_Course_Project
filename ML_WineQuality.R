#Libraries
library(chemometrics)
library(DMwR)
library(FactoMineR)
library(missForest)
library(mice)
library(fpc)
library(ggplot2)
library(gridExtra)
library(plyr) 
library(FactoMineR)
library(factoextra)
library(rpart.plot)
library(dplyr)
library(cowplot)
library(missForest)
library(Boruta)



################ Pre-processing of the data ################

##remove all variables from thr list
rm(list=ls())

#Read the data
redWine <- read.table("winequality-red.csv", header=TRUE, sep=";") #red wine
whiteWine <- read.table("winequality-white.csv", header=TRUE, sep=";") #white wine

#creating a column 'red' for the red wine and 'white' for the white wine
red_wine_type <- data.frame("wine_type" = c('red'))
white_wine_type <- data.frame("wine_type" = c('white'))

#adding the columns to each of the datasets
redWine <- cbind(redWine, red_wine_type)
whiteWine <- cbind(whiteWine, white_wine_type)

#Merging the two dataset 
AllWineData <- rbind(redWine, whiteWine)

##Understanding the data##

#red Wine
str(redWine)
summary(redWine)
#white wine
str(whiteWine)
summary(whiteWine)
#All wine dataset
str(AllWineData)
summary(AllWineData)

##Detecting the missing Values##
apply(AllWineData, 2, function(x) sum(is.na(x)))

#visualizing the missing values
md.pattern(AllWineData)


###detecting outliers###
boxplot(AllWineData)

plot_theme = theme_classic() + theme(plot.title = element_text(hjust = 0.5, size = 14,face = 'bold'),
                                     axis.title.x = element_text(size = 14),
                                     axis.title.y = element_text(size = 14),
                                     axis.text.x  = element_text(size = 12),
                                     axis.text.y  = element_text(size = 12))
#Continuous Univariate Variables
CUVars = function(yfeature, ylabel) {
  
  ggplot(AllWineData, aes(x = "", y = yfeature)) + geom_boxplot(fill = "#0000FF", outlier.colour = "red", outlier.shape = 1) +
    stat_boxplot(geom = "errorbar", width = 0.5) + labs( y = ylabel, title = paste(ylabel, "Distribution")) +plot_theme
}

p1 = CUVars(AllWineData$fixed.acidity, "Fixed Acidity" )
p2 = CUVars(AllWineData$volatile.acidity, "Volatile Acidity" )
p3 = CUVars(AllWineData$citric.acid, "Citric Acid")
p4 = CUVars(AllWineData$residual.sugar, "Residual Sugar" )
p5 = CUVars(AllWineData$chlorides, "Chlorides")
p6 = CUVars(AllWineData$free.sulfur.dioxide, "Free Sultur Dioxide")
p7 = CUVars(AllWineData$total.sulfur.dioxide, "Total Sulfur Dioxide")
p8 = CUVars(AllWineData$density, "Density")
p9 = CUVars(AllWineData$pH, "pH")
p10 = CUVars(AllWineData$sulphates, "Sulphates")
p11 = CUVars(AllWineData$alcohol, "Alcohol")

plot_grid(p1, p2, p3, p4, p5, p6, p7, p8, p9 ,p10, p11)

#Multivariate Outlier Detection using Maholanobis (Robust) Distance
MDoutliers <- Moutlier(AllWineData[,1:12], quantile = 0.975, plot = TRUE)

#Mahalanobis robust distance (MRD)
MRD_index_ordered <- order(MDoutliers$rd, decreasing=T)
round(MDoutliers$rd[MRD_index_ordered][1:4],3)
index <- seq(1:length(AllWineData[,1]))
MRD_df <- cbind.data.frame(index, round(MDoutliers$rd,3), rownames(AllWineData))

#plot
MRD_plot <- plot(MDoutliers$rd, pch="o", cex=1, main="Potential MRD outliers\n by Mahalanobis robust distance",ylab="MRD Rank") 
abline(h = MDoutliers$cutoff, col="red")  # add cutoff line


#treating Outliers
oFixedAcidity = which(AllWineData$fixed.acidity %in% boxplot.stats(AllWineData$fixed.acidity)$out)
oVolatiteAcidity = which(AllWineData$volatile.acidity %in% boxplot.stats(AllWineData$volatile.acidity)$out)
oCitricAcid = which(AllWineData$citric.acid %in% boxplot.stats(AllWineData$citric.acid)$out)
oResidualSugar = which(AllWineData$residual.sugar %in% boxplot.stats(AllWineData$residual.sugar)$out)
oChlorides = which(AllWineData$chlorides %in% boxplot.stats(AllWineData$chlorides)$out)
oFreeSulfurDioxide = which(AllWineData$free.sulfur.dioxide %in% boxplot.stats(AllWineData$free.sulfur.dioxide)$out)
oTotalsulfurDioxide = which(AllWineData$total.sulfur.dioxide %in% boxplot.stats(AllWineData$total.sulfur.dioxide)$out)
oDensity = which(AllWineData$density %in% boxplot.stats(AllWineData$density)$out)
opH = which(AllWineData$pH %in% boxplot.stats(AllWineData$pH)$out)
oSulphates = which(AllWineData$sulphates %in% boxplot.stats(AllWineData$sulphates)$out)
oAlcohol = which(AllWineData$alcohol %in% boxplot.stats(AllWineData$alcohol)$out)

#setting outliers to the value NA
AllWineData[oFixedAcidity, ]$fixed.acidity = NA
AllWineData[oVolatiteAcidity, ]$volatile.acidity = NA
AllWineData[oCitricAcid, ]$citric.acid = NA
AllWineData[oResidualSugar, ]$residual.sugar = NA
AllWineData[oChlorides, ]$chlorides = NA
AllWineData[oFreeSulfurDioxide, ]$free.sulfur.dioxide = NA
AllWineData[oTotalsulfurDioxide, ]$total.sulfur.dioxide = NA
AllWineData[oDensity, ]$density = NA
AllWineData[opH, ]$pH = NA
AllWineData[oSulphates, ]$sulphates= NA
AllWineData[oAlcohol, ]$alcohol= NA

summary(AllWineData)

#Using missForrest to treat the NA's
AllWineData = missForest(AllWineData)
AllWineData = AllWineData$ximp
sum(is.na(AllWineData))

#testing the data after treating outliers
t1 = CUVars(AllWineData$fixed.acidity, "Fixed Acidity" )
t2 = CUVars(AllWineData$volatile.acidity, "Volatile Acidity" )
t3 = CUVars(AllWineData$citric.acid, "Citric Acid")
t4 = CUVars(AllWineData$residual.sugar, "Residual Sugar" )
t5 = CUVars(AllWineData$chlorides, "Chlorides")
t6 = CUVars(AllWineData$free.sulfur.dioxide, "Free Sultur Dioxide")
t7 = CUVars(AllWineData$total.sulfur.dioxide, "Total Sulfur Dioxide")
t8 = CUVars(AllWineData$density, "Density")
t9 = CUVars(AllWineData$pH, "pH")
t10 = CUVars(AllWineData$sulphates, "Sulphates")
t11 = CUVars(AllWineData$alcohol, "Alcohol")

plot_grid(t1, t2, t3, t4, t5, t6, t7, t8, t9 ,t10, t11)

############# End of Pre-processing of the data #############

############# Feature Selection #############

# Perform Boruta search
boruta_output <- Boruta(wine_type ~ ., data=na.omit(AllWineData), doTrace=0)  
names(boruta_output)
# Get significant variables including tentatives
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  
# Do a tentative rough fix
roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

# Variable Importance Scores
imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
imps2[order(-imps2$meanImp), ]  # descending sort

# Plot variable importance
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  




