
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