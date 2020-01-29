### Ubiqum Code Academy_4th Chapter.Big data
### title: Big Data_1.Iphone_Explore dataset and develop models to predict sentiment
### author: Sunin Choi
### date: "10/11/2019"


## 2. Get Started_Explore the Data
#1. Download dataset
library(tidyverse)
library(ggplot2)
library(funModeling)
library(corrplot)
library(dplyr)
library(ggplot2)
library(readxl)
library(readr)
library(plotly)
library(RColorBrewer)
library(Hmisc)
library(earth)
library(caret)
library(reshape)
library(caretTheme)
library(DMwR)


iphone <- read.csv("C:/Users/sunny/Desktop/6.AWS/4_AWS_SentimentAnalysis/1.Documents/iphone.csv")


#2. Basic Exploring
str(iphone) # 12973 obs. 59 variables
summary(iphone)

iphone <- as.data.frame(iphone)

is.vector(iphone)
is.data.frame(iphone$iphonesentiment)

plot_ly(iphone, x= ~iphone$iphonesentiment, type='histogram')

sum(is.na(iphone))


## 3. Preprocessing & Feature Selection
# 1. Examine Correlation
options(max.print = 1000000)
# create a new data set and remove features highly correlated with the dependant 
iphoneCOR <- iphone
iphoneCOR$featureToRemove <- NULL


M <- cor(iphone)
corrplot(M[,1:10], method = "number")

res2 <- rcorr(as.matrix(iphone))
res2
# Extract the correlation coefficients
res2$r
# Extract p-values
res2$P


# 2. Examine Feature Variance
#nearZeroVar() with saveMetrics = TRUE 
#returns an object containing a table including: 
#frequency ratio, percentage unique, zero variance and near zero variance 
nzvMetrics <- nearZeroVar(iphone, saveMetrics = TRUE)
nzvMetrics

# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv <- nearZeroVar(iphone, saveMetrics = FALSE) 
nzv

# create a new data set and remove near zero variance features
iphoneNZV <- iphone[,-nzv]
str(iphoneNZV)


# 3. Recursive Feature Elimination
# Let's sample the data before using RFE
set.seed(123)
iphoneSample <- iphone[sample(1:nrow(iphone), 700, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 1,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 

startingTime <- Sys.time()
rfeipResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)
endTime <- Sys.time()
totalTime <- endTime - startingTime

# Get results
rfeipResults

#sample 100: iphone, htcphone, iphonedisneg, iphoneperpos, samsunggalaxy
#sample 500: iphone, samsunggalaxy, iphonedispos, iphonedisunc, googleandroid
#sample 700: iphone, samsunggalaxy, iphonedispos, iphonedisneg, googleandroid

# Plot results
plot(rfeipResults, type=c("g", "o"))

# create new data set with rfe recommended features
iphoneRFE <- iphone[,predictors(rfeipResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphone$iphonesentiment

# review outcome
str(iphoneRFE)

# Results after preprocessing
# iphonecor
# iphoneNZV 12973 obs 12 var
# iphoneRFE 12973 obs 20 var / 700sam- 15var

# Correlation matrix
correlation_table(iphoneRFE, "iphonesentiment")
var_rank_info(iphoneRFE, "iphonesentiment")

# create a new dataset that will be used for recoding sentiment
ipRC <- iphoneRFE
str(ipRC)

# recode sentiment to combine factor levels 0 & 1 and 4 & 5
ipRC$iphonesentiment <- recode(ipRC$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

ipRC %>% 
  group_by(iphonesentiment) %>% 
  count(iphonesentiment)
#1-2352, 2-454, 3-1188, 4-8979

# inspect results
summary(ipRC)
str(ipRC)

# make galaxysentiment a factor
ipRC$iphonesentiment <- as.factor(ipRC$iphonesentiment)


## 3. Model Development and Evaluation

#1. createDataPartition: spliting the data into two groups
set.seed(369)
trainiphone <- createDataPartition (y = ipRC$iphonesentiment, p = .7, list = FALSE)

trainIP <- ipRC[trainiphone,]
testIP <- ipRC[-trainiphone,]

IPcontrol <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 1, sampling = "smote")

# 2.Train a model with several methods
a <- c("ranger","C5.0", "kknn", "svmLinear")

compare_model <- c()

for(i in a) {
  
  model <- train(iphonesentiment ~., data = trainIP, method = i, trControl =IPcontrol)
  
  pred <- predict(model, newdata = testIP)
  
  pred_metric <- postResample(testIP$iphonesentiment, pred)
  
  compare_model <- cbind(compare_model, pred_metric)
  
}

colnames(compare_model) <- a

compare_model

#            ranger      C5.0      kknn svmLinear(sam700/withoutsample-down)
#Accuracy 0.8483290 0.8483290 0.3771208 0.7760925
#Kappa    0.6246325 0.6243177 0.1515384 0.4153479
#            ranger      C5.0      kknn svmLinear(sam700/withsample-down)
#Accuracy 0.7709512 0.7390746 0.3025707 0.7113111
#Kappa    0.5100812 0.4678653 0.1324391 0.3341544
#            ranger      C5.0      kknn svmLinear(sam700/withsample-smote)
#Accuracy 0.8123393 0.8149100 0.7473008 0.7678663
#Kappa    0.5597752 0.5625015 0.4683687 0.3969723


#2. Structure table for plots
compare_model_melt <- melt(compare_model, varnames = c("metric", "model"))
compare_model_melt <- as_data_frame(compare_model_melt)
compare_model_melt

ggplot(compare_model_melt, aes(x=model, y=value))+
  geom_col()+
  facet_grid(metric~., scales="free")
#without sample - all models are similar
#with sample - knn is worse than others - because the raw data was unequally distributed



## C5.0 Modeling

library(rpart)
library(rpart.plot)
library(ranger)
library(h20)


dim(trainIP)
dim(testIP)
prop.table(table(trainIP$iphonesentiment))
#1          2          3          4 
#0.16140042 0.03501046 0.09159969 0.71198943 
prop.table(table(testIP$iphonesentiment))
#         1          2          3          4 
#0.16118252 0.03496144 0.09151671 0.71233933

dtree_ip <- train(iphonesentiment~., 
                      data = trainIP, 
                      preProc = c("center", "scale"),
                      method = "rpart", 
                      trControl=IPcontrol, 
                      tuneLength = 8)
dtree_ip

rpip<- rpart(iphonesentiment~., data = trainIP, 
                  method = 'class')
rpart.plot(rpip, extra = 100)

# make a prediction
predict_rpip <- predict(rpip, testIP, type = 'class')
# measure performance
table_rpip <- table(testIP$iphonesentiment, predict_rpip)
table_rpip
accuracy_Test <- sum(diag(table_rpip)) / sum(table_rpip)
print(paste('accuracy for test', accuracy_Test))
#"accuracy for test 0.748586118251928"



rfip <- train(iphonesentiment~., 
               data = trainIP, 
               preProc = c("center", "scale"),
               method = "ranger", 
               trControl=IPcontrol,
               tuneLength = 8)
rfip

#mtry  splitrule   Accuracy   Kappa    
#3    gini        0.8474081  0.6179600
#5    gini        0.8483989  0.6226348
#7    gini        0.8474084  0.6211705
#The final values used for the model were mtry = 5, splitrule = gini and min.node.size = 1.
#2    gini        0.8086594  0.5084093 (smote)
#2    extratrees  0.7856530  0.4479344
#3    gini        0.8308951  0.5823077
#3    extratrees  0.8022766  0.5071614
#5    gini        0.8219764  0.5728260
#5    extratrees  0.8225304  0.5715853
#The final values used for the model were mtry = 3, splitrule = gini and min.node.size = 1.
#2    gini        0.7734257  0.5094409 (down)
#2    extratrees  0.7485462  0.4490629
#3    gini        0.7546022  0.4888911
#3    extratrees  0.7493173  0.4697897
#The final values used for the model were mtry = 2, splitrule = gini and min.node.size = 1.

varImp(rfip, scale = FALSE) 
varImp(rfip, scale = FALSE, importance = TRUE)

# make a prediction
predict_rfip <- predict(rfip, testIP)
confusionMatrix(predict_rfip, testIP$iphonesentiment)
postResample(testIP$iphonesentiment, predict_rfip)
#Accuracy 0.8529563, Kappa 0.6344239
#Accuracy 0.8321337, Kappa 0.5882866 (smote)
#Accuracy 0.7681234 0.5063658(down)
 
#          Reference
#Prediction    1    2    3    4
#         1  393    2    1   10
#         2    0   17    0    0
#         3    2    1  235   10
#         4  310  116  120 2673

#   (smote)Reference
#Prediction    1    2    3    4
#         1  380    1    1   17
#         2    6   19    7   23
#         3    9    2  211   26
#         4  310  114  137 2627

#    (down)Reference
#Prediction    1    2    3    4
#         1  387    2    1   19
#         2   21   32    6  219
#         3   22    5  242  128
#         4  275   97  107 2327

# measure performance
table_rfip <- table(testIP$iphonesentiment, predict_rfip)
table_rfip

accuracy_Test <- sum(diag(table_rfip)) / sum(table_rfip)
print(paste('accuracy for test', accuracy_Test))
# "accuracy for test 0.852956298200514"

saveRDS(rfip, "3.Models/rfip.rds")



# Final Prediction with the large metrics
lgmatrixip <- read.csv("1.Documents/largematrix.csv")

lgmatrixip$iphonesentiment <- NA

namesip <- colnames(ipRC)

lgmatrixfinalip <- select(lgmatrix, namesip)

finalpredictionip <- predict(rfip, newdata = lgmatrixfinalip)

table(finalpredictionip)

#1     2     3     4 
#12344  1543  1012 11784 

finalip <- as.data.frame(finalpredictionip)
finalgal <- as.data.frame(finalpredictiongal)

ipgalpredict <- cbind(finalpredictionip, finalpredictiongal)
ipgalpredict <- as.data.frame(ipgalpredict)
ipgalpredict$finalpredictionip <- as.factor(ipgalpredict$finalpredictionip)
ipgalpredict$finalpredictiongal <- as.factor(ipgalpredict$finalpredictiongal)


ggplot(ipgalpredict, aes(x=finalpredictionip))+
  geom_histogram(color="darkblue", fill="lightblue")

ggplot(ipgalpredict, aes(x=finalpredictionip))+
  geom_histogram(color="lightblue", fill="darkblue")

ipgalpredict %>% 
  ggplot(aes(x=finalpredictionip, y=finalpredictiongal))+
  geom_histogram(color="darkblue", fill="lightblue")

library(reshape2)
ipgal = melt(ipgalpredict[,1:2])
ggplot(data = ipgal) +
  geom_histogram(aes(x = value, y=(..count..)/sum(..count..), fill=variable), 
                 alpha=0.3, binwidth=2, position="identity")


plot_num(ipgalpredict)
library(dplyr)
freq(ipgalpredict$finalpredictionip)
freq(ipgalpredict$finalpredictiongal)

           
           