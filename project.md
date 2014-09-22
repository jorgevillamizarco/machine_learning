---
title: "Machine Learning Project"
author: "Jorgevillamizarco"
date: "Thursday, September 20, 2014"
output: html_document
---

##Executive summary

The purpose of the present document is to analyze data captured by Groupware for HAR (Human Activity Recognition) experiments and propose a machine learning algorithm for “classe” variable prediction. Prediction model will be tested with 20 scenarios.

##Getting and Cleaning Data

First we need to download all necessary data and load it removing all missing values using the following code lines:


```r
traindata <- read.table("./pml-training.csv", sep=",", na.strings = c("", "NA", "#DIV/0!"), header=TRUE)
testdata <- read.table("./pml-testing.csv", sep=",", na.strings = c("", "NA", "#DIV/0!"), header=TRUE)
```

traingdata dataset will be used for model creation, training, error calculation and model validation, while testdata dataset will be used for prediction.
With a first look over traindata we can see that there’s a lot of variables with missing values, so in order to get a simpler dataset it’s necessary to remove those variables, first calculating the number of NA’s for each column and then removing all columns with 90% or more of NA’s:



```r
nas <- apply(traindata, 2, function(x) sum(is.na(x)))
nas < 0.9*nrow(traindata)
```

```
##                        X                user_name     raw_timestamp_part_1 
##                     TRUE                     TRUE                     TRUE 
##     raw_timestamp_part_2           cvtd_timestamp               new_window 
##                     TRUE                     TRUE                     TRUE 
##               num_window                roll_belt               pitch_belt 
##                     TRUE                     TRUE                     TRUE 
##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
##                     TRUE                     TRUE                    FALSE 
##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
##                    FALSE                    FALSE                    FALSE 
##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
##                    FALSE                    FALSE                    FALSE 
##           max_picth_belt             max_yaw_belt            min_roll_belt 
##                    FALSE                    FALSE                    FALSE 
##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
##                    FALSE                    FALSE                    FALSE 
##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
##                    FALSE                    FALSE                    FALSE 
##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
##                    FALSE                    FALSE                    FALSE 
##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
##                    FALSE                    FALSE                    FALSE 
##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
##                    FALSE                    FALSE                    FALSE 
##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
##                     TRUE                     TRUE                     TRUE 
##             accel_belt_x             accel_belt_y             accel_belt_z 
##                     TRUE                     TRUE                     TRUE 
##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
##                     TRUE                     TRUE                     TRUE 
##                 roll_arm                pitch_arm                  yaw_arm 
##                     TRUE                     TRUE                     TRUE 
##          total_accel_arm            var_accel_arm             avg_roll_arm 
##                     TRUE                    FALSE                    FALSE 
##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
##                    FALSE                    FALSE                    FALSE 
##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
##                    FALSE                    FALSE                    FALSE 
##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
##                    FALSE                    FALSE                     TRUE 
##              gyros_arm_y              gyros_arm_z              accel_arm_x 
##                     TRUE                     TRUE                     TRUE 
##              accel_arm_y              accel_arm_z             magnet_arm_x 
##                     TRUE                     TRUE                     TRUE 
##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
##                     TRUE                     TRUE                    FALSE 
##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
##                    FALSE                    FALSE                    FALSE 
##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
##                    FALSE                    FALSE                    FALSE 
##            max_picth_arm              max_yaw_arm             min_roll_arm 
##                    FALSE                    FALSE                    FALSE 
##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
##                    FALSE                    FALSE                    FALSE 
##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
##                    FALSE                    FALSE                     TRUE 
##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
##                     TRUE                     TRUE                    FALSE 
##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
##                    FALSE                    FALSE                    FALSE 
##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
##                    FALSE                    FALSE                    FALSE 
##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
##                    FALSE                    FALSE                    FALSE 
##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
##                    FALSE                    FALSE                    FALSE 
## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
##                    FALSE                    FALSE                     TRUE 
##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
##                    FALSE                    FALSE                    FALSE 
##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
##                    FALSE                    FALSE                    FALSE 
##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
##                    FALSE                    FALSE                    FALSE 
##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
##                    FALSE                     TRUE                     TRUE 
##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
##                     TRUE                     TRUE                     TRUE 
##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
##                     TRUE                     TRUE                     TRUE 
##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
##                     TRUE                     TRUE                     TRUE 
##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
##                     TRUE                    FALSE                    FALSE 
##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
##                    FALSE                    FALSE                    FALSE 
##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
##                    FALSE                    FALSE                    FALSE 
##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
##                    FALSE                    FALSE                    FALSE 
##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
##                    FALSE                    FALSE                    FALSE 
##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
##                    FALSE                     TRUE                    FALSE 
##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
##                    FALSE                    FALSE                    FALSE 
##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
##                    FALSE                    FALSE                    FALSE 
##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
##                    FALSE                    FALSE                    FALSE 
##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
##                     TRUE                     TRUE                     TRUE 
##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
##                     TRUE                     TRUE                     TRUE 
##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
##                     TRUE                     TRUE                     TRUE 
##                   classe 
##                     TRUE
```

```r
traindata <- traindata[,(nas < 0.9*nrow(traindata))]
```


First 8 columns are removed too because they have information not necessaire for model creation and finally “classe” variable is converted to factor:


```r
traindata <- traindata[, 8:dim(traindata)[2]]
traindata$classe <- as.factor(traindata$classe)
```

For testdata dataset it’s necessary to remove all variables that were removed from traindata dataset because for next steps we’ll need those two dataset with equal lengths:


```r
defcols <- names(traindata)
testdata <- testdata[, (colnames(testdata) %in% defcols)]
```

##Model Creation

In this step, we are going to create, train, test and calculate errors, so first we need to subset traindata into training (train variable) and testing (test variable) sets for posterior cross-validation:


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
trainingIndex  <- createDataPartition(traindata$classe, p=.60, list=FALSE)
train <- traindata[ trainingIndex,]
test  <- traindata[-trainingIndex,]
```

Prediction model will be created using random forest methodology, the reason for this is that this methodology offers a better accuracy by de-correlating individual random trees and it gives the variable importance which is very useful to detect the most influential variables in model in order to simplify it. Seed is set to a given value for results reproduction, randomForest function is used with “train” subset previously obtained, “classe” as the outcome, “ntree” for setting the number of trees to create, “nodes” to specify the minimum number of variables per each node of the tree and “importance” set in True for variable importance calculation:


```r
set.seed(10)
model <- randomForest(train[,-53], train$classe, ntree=100, nodesize=10, importance=T)
```

Predict and confusionMatrix functions are used to test model and obtain cross-validation results. Prediction result is assigned to “pred” variable, then this one is used in confusionMatrix function to obtain the sample error:


```r
pred <- predict(model, test)
confusionMatrix(pred,test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2225   11    0    0    0
##          B    0 1501   20    0    0
##          C    5    6 1343   28    6
##          D    1    0    5 1258   10
##          E    1    0    0    0 1426
## 
## Overall Statistics
##                                        
##                Accuracy : 0.988        
##                  95% CI : (0.985, 0.99)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.985        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.997    0.989    0.982    0.978    0.989
## Specificity             0.998    0.997    0.993    0.998    1.000
## Pos Pred Value          0.995    0.987    0.968    0.987    0.999
## Neg Pred Value          0.999    0.997    0.996    0.996    0.998
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.171    0.160    0.182
## Detection Prevalence    0.285    0.194    0.177    0.162    0.182
## Balanced Accuracy       0.997    0.993    0.987    0.988    0.994
```

As we can see, the accuracy of the model is more than 99%, and Kappa for statistical relation between model and test subset is 0.98, so, as we expected we have a pretty accurate and low error model.
Using “importance” function we can see which the most influential variables in the model are, the using “varImpPlot” we can see it graphically:


```r
sort(importance(model)[,1], decreasing=T)
```

```
##             yaw_belt    magnet_dumbbell_z            roll_belt 
##               20.122               18.711               17.598 
##    magnet_dumbbell_y           pitch_belt        pitch_forearm 
##               16.491               13.817               13.511 
##         roll_forearm    magnet_dumbbell_x     magnet_forearm_z 
##               12.062               11.502               10.597 
##             roll_arm              yaw_arm     accel_dumbbell_y 
##               10.563               10.489               10.318 
##     gyros_dumbbell_y     magnet_forearm_y        magnet_belt_z 
##               10.110                9.934                9.491 
##        roll_dumbbell         magnet_arm_z      accel_forearm_z 
##                9.431                9.283                9.112 
##         accel_belt_z        magnet_belt_y          yaw_forearm 
##                9.102                9.065                8.995 
##          gyros_arm_y        magnet_belt_x      accel_forearm_x 
##                8.658                8.615                8.444 
##         yaw_dumbbell     accel_dumbbell_z         magnet_arm_x 
##                8.233                8.078                7.948 
##          accel_arm_y          accel_arm_x      gyros_forearm_y 
##                7.895                7.867                7.830 
##         gyros_belt_x total_accel_dumbbell         gyros_belt_z 
##                7.760                7.640                7.633 
##      accel_forearm_y  total_accel_forearm            pitch_arm 
##                7.600                7.478                7.168 
##     accel_dumbbell_x          gyros_arm_x     total_accel_belt 
##                7.137                6.805                6.741 
##     gyros_dumbbell_z     magnet_forearm_x          gyros_arm_z 
##                6.637                6.612                6.405 
##     gyros_dumbbell_x      gyros_forearm_x         magnet_arm_y 
##                6.205                5.812                5.728 
##      gyros_forearm_z       pitch_dumbbell         accel_belt_x 
##                5.698                5.577                5.537 
##          accel_arm_z         accel_belt_y         gyros_belt_y 
##                5.520                5.163                5.126 
##      total_accel_arm 
##                3.675
```

```r
varImpPlot(model, type=1)
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8.png) 

This is very helpful because we can simplify our model to a newer one in where we just use the most influential variables in order to have smaller and faster decision trees.

##Prediction

Prediction will be performed using the “predict” function with model and testdata (20 cases to be predicted) as inputs:


```r
result <- predict(model, newdata=testdata)
result
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

In order to submit result as text files to prediction assignment it’s necessary to use the next function and call:


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(result)
```

Results uploading give a 20/20 score, confirming the accuracy and quality of the model.
