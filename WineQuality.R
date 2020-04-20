#Libraries
library(chemometrics)
library(DMwR)
library(FactoMineR)
library(missForest)
library(mice)
library(fpc)
library(ggplot2)
library(gridExtra)


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

#Detecting the missing Values
apply(AllWineData, 2, function(x) sum(is.na(x)))

#visualizing the missing values
md.pattern(AllWineData)


