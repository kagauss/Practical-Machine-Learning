Practical Machine Learning Course Project
================
kagauss
November 30, 2017

Overview
========

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Read in training and test datasets
----------------------------------

The training data will be partitioned into two seperate sets, training and validation.

The training data will be used to train multiple machine learning models and the validation set will allow testing with the trained model.

``` r
#read in data
training_data <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
test <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.4.2

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.4.2

``` r
#preprocessing
training_data[,-c(1:7, 160)] <- sapply(training_data[,-c(1:7, 160)], as.numeric)
test[,-c(1:7, 160)] <- sapply(test[,-c(1:7, 160)], as.numeric)

#subset columns of interest
training_data <- training_data[8:160]
test <- test[8:160]
```

There are many columns within the dataset that are not of use to our analysis. The *first 7 columns* can be *omitted* as they have no numeric data of interest to our ultimate predications in *"classe"*.

``` r
#check variables in the test set that are NA
names(test[,is.na(apply(test[!names(test) %in% c("classe")],2,sum))])
```

    ##   [1] "kurtosis_roll_belt"       "kurtosis_picth_belt"     
    ##   [3] "kurtosis_yaw_belt"        "skewness_roll_belt"      
    ##   [5] "skewness_roll_belt.1"     "skewness_yaw_belt"       
    ##   [7] "max_roll_belt"            "max_picth_belt"          
    ##   [9] "max_yaw_belt"             "min_roll_belt"           
    ##  [11] "min_pitch_belt"           "min_yaw_belt"            
    ##  [13] "amplitude_roll_belt"      "amplitude_pitch_belt"    
    ##  [15] "amplitude_yaw_belt"       "var_total_accel_belt"    
    ##  [17] "avg_roll_belt"            "stddev_roll_belt"        
    ##  [19] "var_roll_belt"            "avg_pitch_belt"          
    ##  [21] "stddev_pitch_belt"        "var_pitch_belt"          
    ##  [23] "avg_yaw_belt"             "stddev_yaw_belt"         
    ##  [25] "var_yaw_belt"             "var_accel_arm"           
    ##  [27] "avg_roll_arm"             "stddev_roll_arm"         
    ##  [29] "var_roll_arm"             "avg_pitch_arm"           
    ##  [31] "stddev_pitch_arm"         "var_pitch_arm"           
    ##  [33] "avg_yaw_arm"              "stddev_yaw_arm"          
    ##  [35] "var_yaw_arm"              "kurtosis_roll_arm"       
    ##  [37] "kurtosis_picth_arm"       "kurtosis_yaw_arm"        
    ##  [39] "skewness_roll_arm"        "skewness_pitch_arm"      
    ##  [41] "skewness_yaw_arm"         "max_roll_arm"            
    ##  [43] "max_picth_arm"            "max_yaw_arm"             
    ##  [45] "min_roll_arm"             "min_pitch_arm"           
    ##  [47] "min_yaw_arm"              "amplitude_roll_arm"      
    ##  [49] "amplitude_pitch_arm"      "amplitude_yaw_arm"       
    ##  [51] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
    ##  [53] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
    ##  [55] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
    ##  [57] "max_roll_dumbbell"        "max_picth_dumbbell"      
    ##  [59] "max_yaw_dumbbell"         "min_roll_dumbbell"       
    ##  [61] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
    ##  [63] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
    ##  [65] "amplitude_yaw_dumbbell"   "var_accel_dumbbell"      
    ##  [67] "avg_roll_dumbbell"        "stddev_roll_dumbbell"    
    ##  [69] "var_roll_dumbbell"        "avg_pitch_dumbbell"      
    ##  [71] "stddev_pitch_dumbbell"    "var_pitch_dumbbell"      
    ##  [73] "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"     
    ##  [75] "var_yaw_dumbbell"         "kurtosis_roll_forearm"   
    ##  [77] "kurtosis_picth_forearm"   "kurtosis_yaw_forearm"    
    ##  [79] "skewness_roll_forearm"    "skewness_pitch_forearm"  
    ##  [81] "skewness_yaw_forearm"     "max_roll_forearm"        
    ##  [83] "max_picth_forearm"        "max_yaw_forearm"         
    ##  [85] "min_roll_forearm"         "min_pitch_forearm"       
    ##  [87] "min_yaw_forearm"          "amplitude_roll_forearm"  
    ##  [89] "amplitude_pitch_forearm"  "amplitude_yaw_forearm"   
    ##  [91] "var_accel_forearm"        "avg_roll_forearm"        
    ##  [93] "stddev_roll_forearm"      "var_roll_forearm"        
    ##  [95] "avg_pitch_forearm"        "stddev_pitch_forearm"    
    ##  [97] "var_pitch_forearm"        "avg_yaw_forearm"         
    ##  [99] "stddev_yaw_forearm"       "var_yaw_forearm"

``` r
#remove these columns from both test set and training set
nas <- names(test[,is.na(apply(test[!names(test) %in% c("classe")],2,sum))])

training_data <- training_data[,!names(training_data) %in% nas]
test <- test[,!names(test) %in% nas]
dim(training_data) ; dim(test)
```

    ## [1] 19622    53

    ## [1] 20 53

Further, all columns in the test data that are **NA** are of no value to our training models and are also **omitted** from the datasets.

The *training data* is split into *two* datasets:

1.  **Training Set (3/4)**
2.  **Validation Set (1/4)**

``` r
inTrain <- createDataPartition(training_data$classe, p = 3/4, list = FALSE)

training <- training_data[inTrain,]
valid <- training_data[-inTrain,]
```

Machine Learning Methods:
-------------------------

The training data will be used in *four* different ML algorithms to assess the *best model* for the data.

1.  Classification
2.  Random Forest
3.  Boosting
4.  Bagging

For *cross-validation* an additional parameter has been added to the *train()* function with a *k=3 fold*. The trainControl method of "cv" defaults to k=5 but it was just to long to compute so the decrease in the fold helped processing time for this project and still maintained highly accurate output.

### Classification

``` r
#Classification Model
fit_rp <- train(classe~., data = training, method = "rpart")
```

    ## Warning: package 'rpart' was built under R version 3.4.2

``` r
model_rp <- predict(fit_rp, valid)
confusionMatrix(valid$classe, model_rp)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1263   20  108    0    4
    ##          B  431  301  217    0    0
    ##          C  373   27  455    0    0
    ##          D  364  147  293    0    0
    ##          E  128  136  238    0  399
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.4931         
    ##                  95% CI : (0.479, 0.5072)
    ##     No Information Rate : 0.5218         
    ##     P-Value [Acc > NIR] : 1              
    ##                                          
    ##                   Kappa : 0.3373         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.4936  0.47702  0.34706       NA  0.99007
    ## Specificity            0.9437  0.84835  0.88867   0.8361  0.88847
    ## Pos Pred Value         0.9054  0.31718  0.53216       NA  0.44284
    ## Neg Pred Value         0.6307  0.91656  0.78859       NA  0.99900
    ## Prevalence             0.5218  0.12867  0.26733   0.0000  0.08218
    ## Detection Rate         0.2575  0.06138  0.09278   0.0000  0.08136
    ## Detection Prevalence   0.2845  0.19352  0.17435   0.1639  0.18373
    ## Balanced Accuracy      0.7186  0.66269  0.61787       NA  0.93927

### Random Forest

``` r
#Random Forest Model
fit_rf <- train(classe~., data = training, method = "rf", trControl = trainControl(method = "cv", number = 3), importance = T)
```

    ## Warning: package 'randomForest' was built under R version 3.4.2

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
model_rf <- predict(fit_rf, valid)
confusionMatrix(valid$classe, model_rf)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    1    0    0    0
    ##          B    2  945    2    0    0
    ##          C    0    2  851    2    0
    ##          D    0    1    9  792    2
    ##          E    0    0    3    0  898
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9951          
    ##                  95% CI : (0.9927, 0.9969)
    ##     No Information Rate : 0.2847          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9938          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9986   0.9958   0.9838   0.9975   0.9978
    ## Specificity            0.9997   0.9990   0.9990   0.9971   0.9993
    ## Pos Pred Value         0.9993   0.9958   0.9953   0.9851   0.9967
    ## Neg Pred Value         0.9994   0.9990   0.9965   0.9995   0.9995
    ## Prevalence             0.2847   0.1935   0.1764   0.1619   0.1835
    ## Detection Rate         0.2843   0.1927   0.1735   0.1615   0.1831
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9991   0.9974   0.9914   0.9973   0.9985

### Boosting

``` r
#Boosting Model
fit_gbm <- train(classe~., data = training, method = "gbm", verbose = F, trControl = trainControl(method = "cv", number = 3))
```

    ## Warning: package 'gbm' was built under R version 3.4.2

    ## Loading required package: survival

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.3

    ## Warning: package 'plyr' was built under R version 3.4.2

``` r
model_gbm <- predict(fit_gbm, valid)
confusionMatrix(valid$classe, model_gbm)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1371   18    4    1    1
    ##          B   27  887   31    0    4
    ##          C    0   17  832    5    1
    ##          D    1    4   26  764    9
    ##          E    1    9    4   12  875
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9643          
    ##                  95% CI : (0.9587, 0.9693)
    ##     No Information Rate : 0.2855          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9549          
    ##  Mcnemar's Test P-Value : 0.0003623       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9793   0.9487   0.9275   0.9770   0.9831
    ## Specificity            0.9932   0.9844   0.9943   0.9903   0.9935
    ## Pos Pred Value         0.9828   0.9347   0.9731   0.9502   0.9711
    ## Neg Pred Value         0.9917   0.9879   0.9839   0.9956   0.9963
    ## Prevalence             0.2855   0.1907   0.1829   0.1595   0.1815
    ## Detection Rate         0.2796   0.1809   0.1697   0.1558   0.1784
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9862   0.9665   0.9609   0.9836   0.9883

### Bagging

``` r
#TreeBag Model
fit_treebag <- train(classe~., data = training, method = "treebag")
```

    ## Warning: package 'ipred' was built under R version 3.4.2

    ## Warning: package 'e1071' was built under R version 3.4.2

``` r
model_treebag <- predict(fit_treebag, valid)
confusionMatrix(valid$classe, model_treebag)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1372   13    4    1    5
    ##          B   13  920   11    1    4
    ##          C    2    7  837    9    0
    ##          D    1    1   13  786    3
    ##          E    0    4    7    5  885
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9788          
    ##                  95% CI : (0.9744, 0.9826)
    ##     No Information Rate : 0.283           
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.9732          
    ##  Mcnemar's Test P-Value : 0.1402          
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9885   0.9735   0.9599   0.9800   0.9866
    ## Specificity            0.9935   0.9927   0.9955   0.9956   0.9960
    ## Pos Pred Value         0.9835   0.9694   0.9789   0.9776   0.9822
    ## Neg Pred Value         0.9954   0.9937   0.9914   0.9961   0.9970
    ## Prevalence             0.2830   0.1927   0.1778   0.1635   0.1829
    ## Detection Rate         0.2798   0.1876   0.1707   0.1603   0.1805
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9910   0.9831   0.9777   0.9878   0.9913

``` r
plot(fit_gbm)
```

![](PracticalMachineLearningCourseProject_files/figure-markdown_github/unnamed-chunk-9-1.png)

### Conclusion

The **random forest** model had the highest accuracy in predicting the classe variable in the validation set with an accuracy of **99.5%**

The **out of sample error** is **.6%**

**Below** are the **final predictions** on the given test data of twenty samples and the predicted output with the *random forest* model.

``` r
finalModel <- predict(fit_rf, test)
paste(test$problem_id, finalModel, sep = ": ")
```

    ##  [1] "1: B"  "2: A"  "3: B"  "4: A"  "5: A"  "6: E"  "7: D"  "8: B" 
    ##  [9] "9: A"  "10: A" "11: B" "12: C" "13: B" "14: A" "15: E" "16: E"
    ## [17] "17: A" "18: B" "19: B" "20: B"
