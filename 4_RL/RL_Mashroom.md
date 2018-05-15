Calssifying mushroon with Rule Learner
================
Mohammed Zakaria

``` r
library(gmodels) # for CrossTable
library(ggplot2)
#install.packages("C50") # for DT algorithm
library(C50)
library(caret)
#install.packages('pROC')
library(pROC)
library(dplyr)
```

Pulling the data: The mushroom data is from Brett Lantz's "Machine Learning with R" a repo for the data is under this [link](https://github.com/mzakariaCERN/Machine-Learning-with-R-datasets/blob/master/mushrooms.csv) and original data can be found in the [UCI Repository](https://archive.ics.uci.edu/ml)
