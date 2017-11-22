# https://spark.rstudio.com/h2o.html
# Step 1. Sparkling Water (H2O) Machine Learning=====

# ML pipeline steps:
# Perform SQL queries through the sparklyr dplyr interface,
# Use the sdf_* and ft_* family of functions to generate new columns, or partition your data set,
# Convert your training, validation and/or test data frames into H2O Frames using the as_h2o_frame function,
# Choose an appropriate H2O machine learning algorithm to model your data,
# Inspect the quality of your model fit, and use it to make predictions with new data.

# install.packages("rsparkling")

library(sparklyr)
library(rsparkling)
library(h2o)
library(dplyr)

sc <- spark_connect("local") #error prompts installing earlier version of spark, ignored.
mtcars_tbl <- copy_to(sc, mtcars, "mtcars", overwrite = T)

##Pre-processing tasks

# Remove all cars with horsepower less than 100,
# Produce a column encoding whether a car has 8 cylinders or not,
# Partition the data into separate training and test data sets,
# Fit a model to our training data set,
# Evaluate our predictive performance on our test dataset.

partitions <- mtcars_tbl %>%
     filter(hp >= 100) %>%
     mutate(cyl8 = cyl == 8) %>%
     sdf_partition(training = 0.5, test = 0.5, seed = 1099)


training <- as_h2o_frame(sc, partitions$training, strict_version_check = FALSE)
test <- as_h2o_frame(sc, partitions$test, strict_version_check = FALSE)

glm_model <- h2o.glm(x = c("wt", "cyl"), 
                     y = "mpg", 
                     training_frame = training,
                     lambda_search = TRUE)

print(glm.model)


library(ggplot2)

# compute predicted values on our test dataset
pred <- h2o.predict(glm_model, newdata = test) #note the predict command and newdata arguments
# convert from H2O Frame to Spark DataFrame
predicted <- as_spark_dataframe(sc, pred, strict_version_check = FALSE)
predicted

# extract the true 'mpg' values from our test dataset
actual <- partitions$test %>%
     select(mpg) %>%
     collect() %>%
     `[[`("mpg")

# produce a data.frame housing our predicted + actual 'mpg' values
data <- data.frame(
     predicted = predicted,
     actual    = actual
)
# a bug in data.frame does not set colnames properly; reset here 
names(data) <- c("predicted", "actual")

# plot predicted vs. actual values
ggplot(data, aes(x = actual, y = predicted)) +
     geom_abline(lty = "dashed", col = "red") +
     geom_point() +
     theme(plot.title = element_text(hjust = 0.5)) +
     coord_fixed(ratio = 1) +
     labs(
          x = "Actual Fuel Consumption",
          y = "Predicted Fuel Consumption",
          title = "Predicted vs. Actual Fuel Consumption"
     )


# Model predictions are acceptable


## Available ML algorithyms:
#      
# h2o.glm	Generalized Linear Model
# h2o.deeplearning	Multilayer Perceptron
# h2o.randomForest	Random Forest
# h2o.gbm	Gradient Boosting Machine
# h2o.naiveBayes	Naive-Bayes
# h2o.prcomp	Principal Components Analysis
# h2o.svd	Singular Value Decomposition
# h2o.glrm	Generalized Low Rank Model
# h2o.kmeans	K-Means Clustering
# h2o.anomaly	Anomaly Detection via Deep Learning Autoencoder

# Transformers
# 
# ft_binarizer	Threshold numerical features to binary (0/1) feature
# ft_bucketizer	Bucketizer transforms a column of continuous features to a column of feature buckets
# ft_discrete_cosine_transform	Transforms a length NN real-valued sequence in the time domain into another length NN real-valued sequence in the frequency domain
# ft_elementwise_product	Multiplies each input vector by a provided weight vector, using element-wise multiplication.
# ft_index_to_string	Maps a column of label indices back to a column containing the original labels as strings
# ft_quantile_discretizer	Takes a column with continuous features and outputs a column with binned categorical features
# ft_sql_transformer	Implements the transformations which are defined by a SQL statement
# ft_string_indexer	Encodes a string column of labels to a column of label indices
# ft_vector_assembler	Combines a given list of columns into a single vector column


# step 2. an example======

# copy to spark
iris_tbl <- copy_to(sc, iris, "iris", overwrite = TRUE)
iris_tbl

# convert to h2o data frame
iris_hf <- as_h2o_frame(sc, iris_tbl, strict_version_check = FALSE)

# k-means model
kmeans_model <- h2o.kmeans(training_frame = iris_hf, 
                           x = 3:4,
                           k = 3,
                           seed = 1)

# print the cluster centers
h2o.centers(kmeans_model)

# print the centroid statistics
h2o.centroid_stats(kmeans_model)

#Principal component analysis PCA

# Use H2O’s Principal Components Analysis (PCA) to perform dimensionality reduction. 
# PCA is a statistical method to find a rotation such that the first coordinate has 
# the largest variance possible, and each succeeding coordinate in turn has the largest 
# variance possible.



pca_model <- h2o.prcomp(training_frame = iris_hf,
                        x = 1:4,
                        k = 4,
                        seed = 1)
print(pca_model)


# Random Forest
# Use H2O’s Random Forest to perform regression or classification on a dataset. 
# We will continue to use the iris dataset as an example for this problem.

# As usual, we define the response and predictor variables using the x and y arguments.
# Since we’d like to do a classification, we need to ensure that the response column is
# encoded as a factor (enum) column.


y <- "Species"
x <- setdiff(names(iris_hf), y)
iris_hf[,y] <- as.factor(iris_hf[,y])

# We can split the iris_hf H2O Frame into a train and test set (the split defaults to 75/25 train/test).

splits <- h2o.splitFrame(iris_hf, seed = 1)
# Then we can train a Random Forest model:
     
     rf_model <- h2o.randomForest(x = x, 
                                  y = y,
                                  training_frame = splits[[1]],
                                  validation_frame = splits[[2]],
                                  nbins = 32,
                                  max_depth = 5,
                                  ntrees = 20,
                                  seed = 1)
     
# Since we passed a validation frame, the validation metrics will be calculated.
# We can retrieve individual metrics using functions such as h2o.mse(rf_model, valid = TRUE).
# The confusion matrix can be printed using the following:
     
     h2o.confusionMatrix(rf_model, valid = TRUE)

#To view the variable importance computed from an H2O model, you can use either the h2o.varimp() or 
     # h2o.varimp_plot() functions:
     
     h2o.varimp_plot(rf_model)

