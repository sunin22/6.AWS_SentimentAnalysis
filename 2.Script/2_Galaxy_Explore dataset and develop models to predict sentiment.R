### Ubiqum Code Academy_4th Chapter.Big data
### title: Big Data_2.Galaxy_Explore dataset and develop models to predict sentiment
### author: Sunin Choi
### date: "01/124/2019"


## 1. Get Started_Explore the Data
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


galaxy <- read.csv("C:/Users/sunny/Desktop/6.AWS/4_AWS_SentimentAnalysis/1.Documents/galaxy.csv")


#2. Basic Exploring
str(galaxy) # 12973 obs. 59 variables
summary(galaxy)

galaxy <- as.data.frame(galaxy)

is.vector(galaxy)
is.data.frame(galaxy$galaxysentiment)

plot_ly(galaxy, x= ~galaxy$galaxysentiment, type='histogram')

sum(is.na(galaxy))


## 3. Preprocessing & Feature Selection
# 1. Examine Correlation
options(max.print = 1000000)
# create a new data set and remove features highly correlated with the dependant 
galCOR <- galaxy
galCOR$featureToRemove <- NULL

M <- cor(galaxy)
corrplot(M[,1:10], method = "number")

res2 <- rcorr(as.matrix(galaxy))
res2
# Extract the correlation coefficients
res2$r
# Extract p-values
res2$P


# 2. Examine Feature Variance
#nearZeroVar() with saveMetrics = TRUE 
#returns an object containing a table including: 
#frequency ratio, percentage unique, zero variance and near zero variance 
galnzvMetrics <- nearZeroVar(galaxy, saveMetrics = TRUE)
galnzvMetrics

# nearZeroVar() with saveMetrics = FALSE returns an vector 
galnzv <- nearZeroVar(galaxy, saveMetrics = FALSE) 
galnzv

# create a new data set and remove near zero variance features
galNZV <- galaxy[,-galnzv]
str(galNZV) #12973 obs 12 var


# 3. Recursive Feature Elimination
# Let's sample the data before using RFE
set.seed(234)
galSample <- galaxy[sample(1:nrow(galaxy), 700, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 1,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
startingTime <- Sys.time()
galrfeResults <- rfe(galSample[,1:58], 
                  galSample$galaxysentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)
endTime <- Sys.time()
totalTime <- endTime - startingTime
totalTime

# Get results
galrfeResults
#The top 5 variables (out of 16):
#iphone, googleandroid, samsunggalaxy, iphonedisunc, htcphone
#with700/iphone, googleandroid, iphonedispos, samsunggalaxy, iphonedisunc

# Plot results
plot(galrfeResults, type=c("g", "o"))


# create new data set with rfe recommended features
galRFE <- galaxy[,predictors(galrfeResults)]
galRFE

# add the dependent variable to iphoneRFE
galRFE$galaxysentiment <- galaxy$galaxysentiment

# review outcome
str(galRFE)

# Results after preprocessing
# galNZV 12973 obs 12 var
# galRFE 12973 obs 17 var / with700sample 24 var

# Correlation matrix
correlation_table(galRFE, "galaxysentiment")
var_rank_info(galRFE, "galaxysentiment")

# create a new dataset that will be used for recoding sentiment
galRC <- galRFE
str(galRC)

# recode sentiment to combine factor levels 0 & 1 and 4 & 5
galRC$galaxysentiment <- recode(galRC$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

galRC %>% 
  group_by(galaxysentiment) %>% 
  count(galaxysentiment)
#1-2093, 2-454, 3-1188, 4-9238

# inspect results
summary(galRC)
str(galRC)

# make galaxysentiment a factor
galRC$galaxysentiment <- as.factor(galRC$galaxysentiment)


## 3. Model Development and Evaluation

#1. createDataPartition: spliting the data into two groups
pacman:: p_load(caret, reshape, ggplot2, dplyr)


set.seed(369)
traingalaxy <- createDataPartition(y = galRC$galaxysentiment, p = .7, list = FALSE)

traingal <- galRC[traingalaxy,]
testgal <- galRC[-traingalaxy,]

galcontrol <- trainControl(method = "repeatedcv", 
                          number = 10, repeats = 1, sampling = "smote")

# 2.Train a model with several methods
a <- c("ranger","C5.0", "kknn", "svmLinear")

compare_model <- c()

for(i in a) {
  
  model <- train(galaxysentiment ~., data = traingal, method = i, trControl = galcontrol)
  
  pred <- predict(model, newdata = testgal)
  
  pred_metric <- postResample(testgal$galaxysentiment, pred)
  
  compare_model <- cbind(compare_model, pred_metric)
  
}

colnames(compare_model) <- a

compare_model
#            ranger      C5.0      kknn svmLinear(without sample)
#Accuracy 0.8732648 0.8694087 0.3827763 0.8143959
#Kappa    0.6757000 0.6632113 0.1640818 0.4993087
#           ranger      C5.0      kknn svmLinear (with700, sample-down)
#Accuracy 0.818509 0.7858612 0.2897172 0.7670951
#Kappa    0.572522 0.5343228 0.1303923 0.4714607
#            ranger      C5.0      kknn svmLinear(with700, sample-smote)
#Accuracy 0.8146530 0.7940874 0.8123393 0.7537275
#Kappa    0.5735619 0.5165844 0.5610159 0.3830726

#2. Structure table for plots
library(reshape2)
compare_model_melt <- melt(compare_model, varnames = c("metric", "model"))
compare_model_melt <- as_data_frame(compare_model_melt)
compare_model_melt

ggplot(compare_model_melt, aes(x=model, y=value))+
  geom_col()+
  facet_grid(metric~., scales="free")


## Random Forest Modeling

library(rsample)
library(randomForest)
library(ranger)



dim(traingal)
dim(testgal)
prop.table(table(traingal$galaxysentiment))
#1          2          3          4 
#0.16140042 0.03501046 0.09159969 0.71198943 
prop.table(table(testgal$galaxysentiment))
#         1          2          3          4 
#0.16118252 0.03496144 0.09151671 0.71233933


rfgal <- train(galaxysentiment~., 
               data = traingal, 
               preProc = c("center", "scale"),
               method = "ranger", 
               trControl=galcontrol,
               tuneLength = 8)
rfgal

#mtry  splitrule   Accuracy   Kappa    
#2    gini        0.8085439  0.5543366 500sample
#2    extratrees  0.7824508  0.4916392
#4    gini        0.7770591  0.5153826
#4    extratrees  0.7649436  0.4899781
#2    gini        0.8027115  0.5394258 700sample(down)
#2    extratrees  0.7773808  0.4855249
#5    gini        0.7832264  0.5237528
#5    extratrees  0.7689080  0.4948242
#5    gini        0.8226363  0.5733859 700sample(smote)
#5    extratrees  0.7982001  0.5228766
#8    gini        0.8004974  0.5465816


varImp(rfgal, scale = FALSE) 
varImp(rfgal, scale = FALSE, importance = TRUE)

# make a prediction
predict_rfgal <- predict(rfgal, testgal)
confusionMatrix(predict_rfgal, testgal$galaxysentiment)
postResample(testgal$galaxysentiment, predict_rfgal)
#Accuracy 0.8223650, Kappa 0.5769497 (down)
#Accuracy 0.8318766, Kappa 0.6026079 (smote)

#    (down)Reference
#Prediction    1    2    3    4
#         1  387    1    3   28
#         2    1   14   46    0
#         3   20    7  209  154
#         4  219  114   98 2589

#   (smote)Reference
#Prediction    1    2    3    4
#         1  388    0    4   20
#         2   20   23    4  136
#         3    7    2  242   32
#         4  212  111  106 2583

# measure performance
table_rfgal <- table(testgal$galaxysentiment, predict_rfgal)
table_rfgal

accuracy_Test <- sum(diag(table_rfgal)) / sum(table_rfgal)
print(paste('accuracy for test', accuracy_Test))
# "accuracy for test 0.822365038560411"

saveRDS(rfgal, "3.Models/rfgal.rds")



# Final Prediction with the large metrics
lgmatrixgal <- read.csv("1.Documents/largematrix.csv")

lgmatrixgal$galaxysentiment <- NA

namesgal <- colnames(galRC)

lgmatrixfinalgal <- select(lgmatrixgal, namesgal)

finalpredictiongal <- predict(rfgal, newdata = lgmatrixfinalgal)


table(finalpredictiongal)
#1     2     3     4 
#12541  1755  1127 11260 
