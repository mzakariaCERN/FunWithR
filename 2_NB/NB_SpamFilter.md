Spam SMS classifier with Naive Bays
================
Mohammed Zakaria

Required packages

``` r
#install.packages("class") # for kNN classification
library(class)
#install.packages("gmodels") # for CrossTable function at the evaluation
library(gmodels)
#install.packages("caret") # for model tuning
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
#install.packages("e1071") # to help with model tuning
library(e1071)
#install.packages("pROC") # to make ROC plots
library(pROC)   
```

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following object is masked from 'package:gmodels':
    ## 
    ##     ci

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

Pulling the data: The cancer data is from Brett Lantz's "Machine Learning with R" a repo for the data is under this link: <https://github.com/mzakariaCERN/Machine-Learning-with-R-datasets/blob/master/wisc_bc_data.csv> and original data can be found under <https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/>

``` r
sms_raw <- read.csv(file="C:/Users/mkzak/Documents/GitHub/FunWithR/FunWithR/2_NB/Data/sms_spam.csv", stringsAsFactors = FALSE)


dim(sms_raw)
```

    ## [1] 5574    2

``` r
str(sms_raw)
```

    ## 'data.frame':    5574 obs. of  2 variables:
    ##  $ type: chr  "ham" "ham" "spam" "ham" ...
    ##  $ text: chr  "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..." "Ok lar... Joking wif u oni..." "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question("| __truncated__ "U dun say so early hor... U c already then say..." ...

``` r
summary(sms_raw)
```

    ##      type               text          
    ##  Length:5574        Length:5574       
    ##  Class :character   Class :character  
    ##  Mode  :character   Mode  :character

We see there are two features. And the feature type has a categorical variables. So we need to convert it to factor

``` r
sms_raw$type <- as.factor(sms_raw$type)

str(sms_raw$type)
```

    ##  Factor w/ 2 levels "ham","spam": 1 1 2 1 1 2 1 1 2 2 ...

``` r
table(sms_raw$type)
```

    ## 
    ##  ham spam 
    ## 4827  747

Add a new chunk by clicking the *Insert Chunk* button on the toolbar or by pressing *Ctrl+Alt+I*.

When you save the notebook, an HTML file containing the code and output will be saved alongside it (click the *Preview* button or press *Ctrl+Shift+K* to preview the HTML file).

The preview shows you a rendered HTML copy of the contents of the editor. Consequently, unlike *Knit*, *Preview* does not run any R code chunks. Instead, the output of the chunk when it was last run in the editor is displayed.
