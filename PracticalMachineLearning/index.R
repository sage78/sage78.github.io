
# This is a write up for Coursera's course Practical Machine Learning and it decribes how
# a machine learning algortihm was developed to predict the manner in which people did
# certain sports execrices by using sensor data collected from devices such as 
# Jawbone Up, Nike FuelBand and Fitbit. More information is available at
# http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

# Data was provided in two datasets: training and testing data

# Building the model included the following phases:
# - General settings
# - Reading data in and splitting it into train and test datasets
# - Transforming the training data
# - Training a machine learning algorithm by using training data
# - Estimating accuracy of the algorithm by using testing data

## General settings

# You should have pml-training.csv and pml-testing.csv data sets in your working directory.
library(caret)
set.seed(12345)


## Reading data in and splitting it into train and test datasets

# Data used for training an algortihm and evaluating it was provided in pml-training.csv. The file
# contains several measurement values which are erroneous, so we treat NA, #DIV/0! and empty values 
# as missing ones. Classe is the outcome to be predicted.
# After reading in the data, We split it in two parts randomly:
# - 80% of the data is used for training an algortihm and 
# - 20% of the data is used for evaluating its accuracy (testing it)
data <- read.csv("pml-training.csv", header=TRUE, na.strings=c("NA", "#DIV/0!", ""))
inTrain <- createDataPartition(y=data$classe, p=0.8, list=FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]

# Source data contains nrow(data) rows and ncol(data) columns in total.


## Transforming the training data

# Training data contains nrow(training) rows and ncol(training) columns. However, a brief study incidated that several
# columns contained mostly NA values. In addition, columns 1-7  contained user information, time stamps, windows 
# etc which are not relevant sensor data for building the model. As a result, the following transformations were
# made:
# - remove columns 1-7
# - remove columns containing NA values
# - remove columns with name starting as total_ (such as total_accel_arm), as they compose data included in other columns
training <- training[-c(1:7)]
training <- training[, apply(training, 2, function(x) !any(is.na(x)))] 
training <- training[, -grep("^total_", colnames(training))]

# After the transformations, training data contained nrow(training) rows and ncol(training) columns. All
# columns are numeric or integer except classe, which is factor and presents the manner in which the 
# subject did the exercise - it is to be predicted.

# nearZeroVar diagnose was used for detecting possible variables that have one unique value 
# (i.e. are zero variance predictors) and predictors that are have both of the following 
# characteristics: they have very few unique values relative to the number of samples and 
# the ratio of the frequency of the most common value to the frequency of the second 
# most common value is large. It did not result in removing variables.
nearZeroVar(training)

# Correlated variables were identified in order to remove duplicate information, which might lead
# into  overfitting a model.
cors <- abs(cor(training[,-grep("classe", colnames(training))]))
diag(cors) <- 0
cors[cors < 0.9] <- NA
cors <- cors[, apply(cors, 2, function(x) !all(is.na(x)))]
cors <- cors[apply(cors, 1, function(x) !all(is.na(x))), ]
cors

# As a result we notice that 
# - roll_belt correlates with accel_belt_y and accel_belt_z
# - pitch_belt correlates with accel_belt_x
# - gyros_dumbbell_x correlates with gyros_dumbbell_z and gyros_forearm_z
# - gyros_arm_x correlates with gyros_arm_y

# As a result, we remove the correlating variables
training <- training[,!(names(training) %in% c("accel_belt_y","accel_belt_z",
                                               "accel_belt_x",
                                               "gyros_dumbbell_z", "gyros_forearm_z",
                                               "gyros_arm_y"))]


## Training a machine learning algorithm by using training data

# As a next step, we train a machine learning algorithm by using the training data. 
# Random forest seems like a good choice thanks to its accuracy.

# Lets use repeated k-fold cross validation, in which the process of splitting the data 
# into k-folds is repeated a number of times. The final model accuracy is taken as 
# the mean from the number of repeats.

# For the sake of processing time, we use 3-fold cross validation with 3 repeats
train_control <- trainControl(method="repeatedcv", number=3, repeats=3)
model <- train(classe ~ ., data=training, trControl=train_control, method="rf", prox = TRUE)

# To avoid overfitting, we then reduce amount of predictor variables to 15 most important
# according to varImpPlot results
varImpPlot(model$finalModel)
varImp <- varImp(model$finalModel)
varImp <- rownames(varImp)[order(varImp$Overall, decreasing=TRUE)][1:15]
training <- training[,(names(training) %in% c(varImp, "classe"))]

# Build a new model by using only these columns are predictors
train_control2 <- trainControl(method="repeatedcv", number=3, repeats=3)
model2 <- train(classe ~ ., data=training, trControl=train_control, method="rf", prox = TRUE)

# The model is summarized below
print(model2)
print(model2$finalModel)


## Estimating accuracy of the algorithm by using testing data

# Use developed model on testing data
predict <- predict(model2, newdata=testing)

# Calculate estimate on out of sample error by using testing data. 
# The real out of sample error is likely to be slighly higher.
confusionMatrix(predict, testing$classe)

# As a conclusion, the model seem to perform rather accurately. We have tried to minimize
# risk of overfitting by reducing amount of variables used in the model.





