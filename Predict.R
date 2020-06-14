library(caret)
library(randomForest)
library(ROCR)
library(gridExtra)
library(nnet)
library(mlr)
library(aod)
library(e1071) #SVM

AllWineData <- read.table("AllWineDataPreProcessed.csv", header=TRUE, sep=";")
names(AllWineData)
plot(as.factor(AllWineData$quality))
title(main = "Quality distribution")
AllWineData$taste <- ifelse(AllWineData$quality < 5, "bad", "good")
AllWineData$taste[AllWineData$quality == 5] <- "normal"
AllWineData$taste[AllWineData$quality == 6] <- "normal"
AllWineData$taste[AllWineData$quality >= 8] <- "excellent"
AllWineData$taste <- as.factor(AllWineData$taste)
plot(AllWineData$taste)
title(main= "Quality Distribution")
index <- createDataPartition(AllWineData$quality, p=0.7, list=FALSE)
wine <- AllWineData[,c(-13)]
train <- wine[index,]
test <- wine[-index,]
N <- nrow(wine)
learn <- sample(1:N, round(2*N/3))  # random indices for the learning set
nlearn <- length(learn)
ntest <- N - nlearn

##RANDOM FOREST
#Find out ntree best
ntrees <- round(10^seq(1,3.2,by=0.2))
rf.results <- matrix (rep(0,2*length(ntrees)),nrow=length(ntrees))
colnames (rf.results) <- c("ntrees", "OOB")
rf.results[,"ntrees"] <- ntrees
rf.results[,"OOB"] <- 0
ii <- 1
for (nt in ntrees)
{ 
  print(nt)
  set.seed(2018)
  model.rf <- randomForest(taste ~ . - quality, data = train, ntree=nt, proximity=FALSE)
  # get the OOB
  rf.results[ii,"OOB"] <- model.rf$err.rate[nt,1]
  
  ii <- ii+1
}
lowest.OOB.error <- as.integer(which.min(rf.results[,"OOB"]))
(ntrees.best <- rf.results[lowest.OOB.error,"ntrees"])
plot(x = rf.results[,1], y=rf.results[,2], ylab = "OOB", xlab = "# trees", type = "o")
title("Estimate error rate")
grid.table(rf.results[,1:2])

#Model
model <- randomForest(taste ~ . - quality, data=train, ntree=ntrees.best,proximity=TRUE, importance=TRUE,
                      keep.forest=TRUE)
varImpPlot(model, main = "Importance of variables")
impvar <- importance(model)
plot(impvar)
prediction <- predict(model, newdata = test)
# What variables are being used in the forest (their total counts)
var <- varUsed(model, by.tree=FALSE, count = TRUE)
var = as.data.frame((var))
rownames(var) = colnames(test[,1:11])
barplot(as.matrix(t(var)), las=2, cex.names = 0.55)
title("Variables being used in the forest")

result <- table(prediction, test$taste)
round(100*(1-sum(diag(result))/sum(result)),2)
#Precision
(precision <- diag(result) / rowSums(result))
#Recall
(recall <- (diag(result) / colSums(result)))
#accuracy
(accuracy <- sum(diag(result)) / sum(result))

##NEURAL NETWORK
trc <- trainControl(method="repeatedcv", number=10, repeats=10)
#Find best size Neural
(sizes <- seq(10,30,by=10))
(decays <- 10^seq(-2, 0, by=0.2))
model.10x10CV <- train(taste ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + 
                          chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH +
                          sulphates + alcohol, data = train, 
                        method='nnet', maxit = 1000, trace = FALSE,
                        tuneGrid = expand.grid(.size=30,.decay=0.01), trControl=trc)
model.10x10CV$bestTune
model.10x10CV$results
#Another way to train the model
learned.nnet  <- nnet(taste ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + 
                         chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH +
                         sulphates + alcohol, data = train, size=30, maxit=1000, trace=F)
plot(learned.nnet, nid=F)
#Prediction
prediction.nnet <- predict(learned.nnet, newdata = test, type="class")
#Results
p2 <- as.factor(prediction.nnet)
t2 <- table(pred=prediction.nnet, truth = test$taste)
(error_rate.test <- 100*(1-sum(diag(t2))/sum(t2)))
(accuracy <- sum(diag(t2)) / sum(t2))
#Precision
(precision <- diag(t2) / rowSums(t2))
#Recall
(recall <- (diag(t2) / colSums(t2)))
plot(prediction.nnet)

## MULTINOMIAL LOGISTIC REGRESSION
#by default bad as a baseline (???) yes
mln <- multinom(taste ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + 
                chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH +
                sulphates + alcohol, data = train)
summary(mln)
(z <- summary(mln)$coefficients/summary(mln)$standard.errors)
# 2-tailed Wald z tests to test significance of coefficients
(p <- (1 - pnorm(abs(z), 0, 1)) * 2)
exp(coef(mln))
#removed variables
mln2 <- multinom(taste ~ volatile.acidity + residual.sugar + free.sulfur.dioxide + total.sulfur.dioxide 
                 + density + sulphates + alcohol, data = train)
summary(mln2)
exp(coef(mln2))
prediction.mln <- predict(mln2, newdata = test)
pred_mln <- as.factor(prediction.mln)
t_mln <- (table(table = pred_mln, truth =test$taste))
(error_mlr.test <- round(100*(1-sum(diag(t_mln))/nrow(test)),2))
(accuracy <- round(100*(sum(diag(t_mln)) / sum(t_mln)),2))


## Support Vector Machines ## (Not working probably needs to be removed)
train.svm.kCV <- function (which.kernel, myC, myG, kCV=10)
{
  for (i in 1:kCV) 
  {  
    train <- AllWineData[folds!=i,] # for building the model (training)
    valid <- AllWineData[folds==i,] # for prediction (validation)
    
    x_train <- train[,1:11]
    t_train <- train[,13]
    
    switch(which.kernel,
           linear={model <- svm(x_train, t_train, type="C-classification", 
                                cost=myC, gamma=myG, kernel="linear", scale = FALSE)},
           poly.2={model <- svm(x_train, t_train, type="C-classification", 
                                cost=myC, gamma=myG, kernel="polynomial", degree=2, coef0=1, scale = FALSE)},
           poly.3={model <- svm(x_train, t_train, type="C-classification", 
                                cost=myC, gamma=myG, kernel="polynomial", degree=3, coef0=1, scale = FALSE)},
           RBF={model <- svm(x_train, t_train, type="C-classification", 
                             cost=myC, gamma=myG, kernel="radial", scale = FALSE)},
           stop("Enter one of 'linear', 'poly.2', 'poly.3', 'radial'"))
    
    x_valid <- valid[,1:11]
    pred <- predict(model,x_valid)
    t_true <- valid[,13]
    
    # compute validation error for part 'i'
    valid.error[i] <- sum(pred != t_true)/length(t_true)
  }
  
  # return average validation error in percentage
  100*sum(valid.error)/length(valid.error)
}

k <- 10 
folds <- sample(rep(1:k, length=N), N, replace=FALSE) 
valid.error <- rep(0,k)

C <- 1
for (C in 10^seq(-2,3)) {
  print("C:")
  print(C)
  # Fit an SVM with linear kernel
  (VA.error.linear <- train.svm.kCV("linear", myC=C))
  print("linear")
  print(VA.error.linear)
  ## Fit an SVM with quadratic kernel 
  (VA.error.poly.2 <- train.svm.kCV("poly.2", myC=C))
  print("poly.2")
  print(VA.error.poly.2)
  ## Fit an SVM with cubic kernel
  (VA.error.poly.3 <- train.svm.kCV("poly.3", myC=C))
  print("poly.3")
  print(VA.error.poly.3)
  ## and finally an RBF Gaussian kernel 
  (VA.error.RBF <- train.svm.kCV ("RBF", myC=C))
  print("RBF")
  print(VA.error.RBF)
}
for (g in 2^seq(-3,4)) {
  print("g:")
  print(g)
  ## Fit an SVM with quadratic kernel 
  (VA.error.poly.2 <- train.svm.kCV("poly.2", myC=0.1, myG=g))
  print("poly.2")
  print(VA.error.poly.2)
}

model <- svm(taste ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + 
               chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH +
               sulphates + alcohol, train, type="C-classification", cost=0.1, gamma=0.125, kernel="polynomial", degree=2, coef0=1, scale = FALSE)

prediction <- predict(model, newdata = test)

result <- table(prediction, test$taste)
round(100*(1-sum(diag(result))/sum(result)),2)
#Precision
(precision <- diag(result) / rowSums(result))
#Recall
(recall <- (diag(result) / colSums(result)))
#accuracy
(accuracy <- sum(diag(result)) / sum(result))