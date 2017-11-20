# Step 1. Installation ==========================
install.packages("sparklyr")

library(sparklyr)
spark_install(version = "2.1.0")
devtools::install_github("rstudio/sparklyr")
# Restart R session

#Step 2. Conncet to local instance =======
library(sparklyr)
sc <- spark_connect(master = "local") # I can see the conncetion on my connections tab

install.packages(c("nycflights13", "Lahman"))

#Step 3. Interacting with Spark connection using Tidyverse======
library(dplyr)
iris_tbl <- copy_to(sc, iris) #upload iris df to remote (spark instance here)
flights_tbl <- copy_to(sc, nycflights13::flights, "flights")
batting_tbl <- copy_to(sc, Lahman::Batting, "batting")

src_tbls(sc)

flights_tbl %>% str() #not a tibble but a spark connection
flights_tbl %>% filter(dep_delay == 2)


delay <- flights_tbl %>% 
     group_by(tailnum) %>%
     summarise(count = n(), dist = mean(distance), delay = mean(arr_delay)) %>%
     filter(count > 20, dist < 2000, !is.na(delay)) %>%
     collect()

delay %>% str() #delay is actually a tibble as collect command was used (query was processed)

library(ggplot2)
ggplot(delay, aes(dist, delay)) +
     geom_point(aes(size = count), alpha = 1/2) +
     geom_smooth() +
     scale_size_area(max_size = 2)

batting_tbl %>%
     select(playerID, yearID, teamID, G, AB:H) %>%
     arrange(playerID, yearID, teamID) %>%
     group_by(playerID) %>%
     filter(min_rank(desc(H)) <= 2 & H > 0)

# Step 4. Using SQL ======

library(DBI)
iris_preview <- dbGetQuery(sc, "SELECT * FROM iris LIMIT 10")
iris_preview #note that query was executed


# Step 5. Machine learning =====

# copy mtcars into spark
mtcars_tbl <- copy_to(sc, mtcars)

# transform our data set, and then partition into 'training', 'test'
partitions <- mtcars_tbl %>%
     filter(hp >= 100) %>%
     mutate(cyl8 = cyl == 8) %>%
     sdf_partition(training = 0.5, test = 0.5, seed = 1099) #sparklyr function

# fit a linear model to the training dataset
fit <- partitions$training %>%
     ml_linear_regression(response = "mpg", features = c("wt", "cyl"))

summary(fit)

# Step 6. Data I/O =====

temp_csv <- tempfile(fileext = ".csv")
temp_parquet <- tempfile(fileext = ".parquet")
temp_json <- tempfile(fileext = ".json")

spark_write_csv(iris_tbl, temp_csv)
iris_csv_tbl <- spark_read_csv(sc, "iris_csv", temp_csv)

spark_write_parquet(iris_tbl, temp_parquet)
iris_parquet_tbl <- spark_read_parquet(sc, "iris_parquet", temp_parquet)

spark_write_json(iris_tbl, temp_json)
iris_json_tbl <- spark_read_json(sc, "iris_json", temp_json)

src_tbls(sc)

# Step 7. Distributed R

rgamma(1,2)

spark_apply(iris_tbl, function(data) {
     data[1:4] + rgamma(1,2)
})


spark_apply(
     iris_tbl,
     function(e) broom::tidy(lm(Petal_Width ~ Petal_Length, e)),
     names = c("term", "estimate", "std.error", "statistic", "p.value"),
     group_by = "Species"
)


# Step 8. Extensions========

# write a CSV 
tempfile <- tempfile(fileext = ".csv")
write.csv(nycflights13::flights, tempfile, row.names = FALSE, na = "")

# define an R interface to Spark line counting
count_lines <- function(sc, path) {
     spark_context(sc) %>% 
          invoke("textFile", path, 1L) %>% 
          invoke("count")
}

# call spark to count the lines of the CSV
count_lines(sc, tempfile)


# Step 9. Table caching =====

tbl_cache(sc, "batting")
tbl_uncache(sc, "batting")


# Step 10. Connection Utils =====

spark_web(sc)

spark_log(sc, n = 10)

# Step 11. Disconect from Spark ======
spark_disconnect(sc)

# Step 11. Using H2O ======

options(rsparkling.sparklingwater.version = "2.1.14")                                                

library(rsparkling)
library(sparklyr)
library(dplyr)
library(h2o)

sc <- spark_connect(master = "local", version = "2.1.0")
mtcars_tbl <- copy_to(sc, mtcars, "mtcars")

mtcars_h2o <- as_h2o_frame(sc, mtcars_tbl, strict_version_check = FALSE)

mtcars_glm <- h2o.glm(x = c("wt", "cyl"), 
                      y = "mpg",
                      training_frame = mtcars_h2o,
                      lambda_search = TRUE)
mtcars_glm


