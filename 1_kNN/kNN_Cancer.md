Tumer Classification with kNN
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

    ## Warning: package 'caret' was built under R version 3.5.1

``` r
#install.packages("e1071") # to help with model tuning
library(e1071)
#install.packages("pROC") # to make ROC plots
library(pROC)   
library(visdat) # better data exploration with vis_dat
```

    ## Warning: package 'visdat' was built under R version 3.5.1

``` r
library(assertr) # mkae assert statements for better data quality
```

    ## Warning: package 'assertr' was built under R version 3.5.1

``` r
library(dplyr)
```

Pulling the data: The cancer data is from Brett Lantz’s “Machine
Learning with R” a repo for the data is under this link:
<https://github.com/mzakariaCERN/Machine-Learning-with-R-datasets/blob/master/wisc_bc_data.csv>
and original data can be found under
<https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/>

``` r
wbcd <- read.csv(file = "C:/Users/mkzak/Documents/GitHub/FunWithR/FunWithR/1_kNN/wisc_bc_data.csv", stringsAsFactors = FALSE)


dim(wbcd)
```

    ## [1] 569  32

``` r
str(wbcd)
```

    ## 'data.frame':    569 obs. of  32 variables:
    ##  $ id                     : int  842302 842517 84300903 84348301 84358402 843786 844359 84458202 844981 84501001 ...
    ##  $ diagnosis              : chr  "M" "M" "M" "M" ...
    ##  $ radius_mean            : num  18 20.6 19.7 11.4 20.3 ...
    ##  $ texture_mean           : num  10.4 17.8 21.2 20.4 14.3 ...
    ##  $ perimeter_mean         : num  122.8 132.9 130 77.6 135.1 ...
    ##  $ area_mean              : num  1001 1326 1203 386 1297 ...
    ##  $ smoothness_mean        : num  0.1184 0.0847 0.1096 0.1425 0.1003 ...
    ##  $ compactness_mean       : num  0.2776 0.0786 0.1599 0.2839 0.1328 ...
    ##  $ concavity_mean         : num  0.3001 0.0869 0.1974 0.2414 0.198 ...
    ##  $ concave.points_mean    : num  0.1471 0.0702 0.1279 0.1052 0.1043 ...
    ##  $ symmetry_mean          : num  0.242 0.181 0.207 0.26 0.181 ...
    ##  $ fractal_dimension_mean : num  0.0787 0.0567 0.06 0.0974 0.0588 ...
    ##  $ radius_se              : num  1.095 0.543 0.746 0.496 0.757 ...
    ##  $ texture_se             : num  0.905 0.734 0.787 1.156 0.781 ...
    ##  $ perimeter_se           : num  8.59 3.4 4.58 3.44 5.44 ...
    ##  $ area_se                : num  153.4 74.1 94 27.2 94.4 ...
    ##  $ smoothness_se          : num  0.0064 0.00522 0.00615 0.00911 0.01149 ...
    ##  $ compactness_se         : num  0.049 0.0131 0.0401 0.0746 0.0246 ...
    ##  $ concavity_se           : num  0.0537 0.0186 0.0383 0.0566 0.0569 ...
    ##  $ concave.points_se      : num  0.0159 0.0134 0.0206 0.0187 0.0188 ...
    ##  $ symmetry_se            : num  0.03 0.0139 0.0225 0.0596 0.0176 ...
    ##  $ fractal_dimension_se   : num  0.00619 0.00353 0.00457 0.00921 0.00511 ...
    ##  $ radius_worst           : num  25.4 25 23.6 14.9 22.5 ...
    ##  $ texture_worst          : num  17.3 23.4 25.5 26.5 16.7 ...
    ##  $ perimeter_worst        : num  184.6 158.8 152.5 98.9 152.2 ...
    ##  $ area_worst             : num  2019 1956 1709 568 1575 ...
    ##  $ smoothness_worst       : num  0.162 0.124 0.144 0.21 0.137 ...
    ##  $ compactness_worst      : num  0.666 0.187 0.424 0.866 0.205 ...
    ##  $ concavity_worst        : num  0.712 0.242 0.45 0.687 0.4 ...
    ##  $ concave.points_worst   : num  0.265 0.186 0.243 0.258 0.163 ...
    ##  $ symmetry_worst         : num  0.46 0.275 0.361 0.664 0.236 ...
    ##  $ fractal_dimension_worst: num  0.1189 0.089 0.0876 0.173 0.0768 ...

``` r
summary(wbcd)
```

    ##        id             diagnosis          radius_mean      texture_mean  
    ##  Min.   :     8670   Length:569         Min.   : 6.981   Min.   : 9.71  
    ##  1st Qu.:   869218   Class :character   1st Qu.:11.700   1st Qu.:16.17  
    ##  Median :   906024   Mode  :character   Median :13.370   Median :18.84  
    ##  Mean   : 30371831                      Mean   :14.127   Mean   :19.29  
    ##  3rd Qu.:  8813129                      3rd Qu.:15.780   3rd Qu.:21.80  
    ##  Max.   :911320502                      Max.   :28.110   Max.   :39.28  
    ##  perimeter_mean     area_mean      smoothness_mean   compactness_mean 
    ##  Min.   : 43.79   Min.   : 143.5   Min.   :0.05263   Min.   :0.01938  
    ##  1st Qu.: 75.17   1st Qu.: 420.3   1st Qu.:0.08637   1st Qu.:0.06492  
    ##  Median : 86.24   Median : 551.1   Median :0.09587   Median :0.09263  
    ##  Mean   : 91.97   Mean   : 654.9   Mean   :0.09636   Mean   :0.10434  
    ##  3rd Qu.:104.10   3rd Qu.: 782.7   3rd Qu.:0.10530   3rd Qu.:0.13040  
    ##  Max.   :188.50   Max.   :2501.0   Max.   :0.16340   Max.   :0.34540  
    ##  concavity_mean    concave.points_mean symmetry_mean   
    ##  Min.   :0.00000   Min.   :0.00000     Min.   :0.1060  
    ##  1st Qu.:0.02956   1st Qu.:0.02031     1st Qu.:0.1619  
    ##  Median :0.06154   Median :0.03350     Median :0.1792  
    ##  Mean   :0.08880   Mean   :0.04892     Mean   :0.1812  
    ##  3rd Qu.:0.13070   3rd Qu.:0.07400     3rd Qu.:0.1957  
    ##  Max.   :0.42680   Max.   :0.20120     Max.   :0.3040  
    ##  fractal_dimension_mean   radius_se        texture_se      perimeter_se   
    ##  Min.   :0.04996        Min.   :0.1115   Min.   :0.3602   Min.   : 0.757  
    ##  1st Qu.:0.05770        1st Qu.:0.2324   1st Qu.:0.8339   1st Qu.: 1.606  
    ##  Median :0.06154        Median :0.3242   Median :1.1080   Median : 2.287  
    ##  Mean   :0.06280        Mean   :0.4052   Mean   :1.2169   Mean   : 2.866  
    ##  3rd Qu.:0.06612        3rd Qu.:0.4789   3rd Qu.:1.4740   3rd Qu.: 3.357  
    ##  Max.   :0.09744        Max.   :2.8730   Max.   :4.8850   Max.   :21.980  
    ##     area_se        smoothness_se      compactness_se      concavity_se    
    ##  Min.   :  6.802   Min.   :0.001713   Min.   :0.002252   Min.   :0.00000  
    ##  1st Qu.: 17.850   1st Qu.:0.005169   1st Qu.:0.013080   1st Qu.:0.01509  
    ##  Median : 24.530   Median :0.006380   Median :0.020450   Median :0.02589  
    ##  Mean   : 40.337   Mean   :0.007041   Mean   :0.025478   Mean   :0.03189  
    ##  3rd Qu.: 45.190   3rd Qu.:0.008146   3rd Qu.:0.032450   3rd Qu.:0.04205  
    ##  Max.   :542.200   Max.   :0.031130   Max.   :0.135400   Max.   :0.39600  
    ##  concave.points_se   symmetry_se       fractal_dimension_se
    ##  Min.   :0.000000   Min.   :0.007882   Min.   :0.0008948   
    ##  1st Qu.:0.007638   1st Qu.:0.015160   1st Qu.:0.0022480   
    ##  Median :0.010930   Median :0.018730   Median :0.0031870   
    ##  Mean   :0.011796   Mean   :0.020542   Mean   :0.0037949   
    ##  3rd Qu.:0.014710   3rd Qu.:0.023480   3rd Qu.:0.0045580   
    ##  Max.   :0.052790   Max.   :0.078950   Max.   :0.0298400   
    ##   radius_worst   texture_worst   perimeter_worst    area_worst    
    ##  Min.   : 7.93   Min.   :12.02   Min.   : 50.41   Min.   : 185.2  
    ##  1st Qu.:13.01   1st Qu.:21.08   1st Qu.: 84.11   1st Qu.: 515.3  
    ##  Median :14.97   Median :25.41   Median : 97.66   Median : 686.5  
    ##  Mean   :16.27   Mean   :25.68   Mean   :107.26   Mean   : 880.6  
    ##  3rd Qu.:18.79   3rd Qu.:29.72   3rd Qu.:125.40   3rd Qu.:1084.0  
    ##  Max.   :36.04   Max.   :49.54   Max.   :251.20   Max.   :4254.0  
    ##  smoothness_worst  compactness_worst concavity_worst  concave.points_worst
    ##  Min.   :0.07117   Min.   :0.02729   Min.   :0.0000   Min.   :0.00000     
    ##  1st Qu.:0.11660   1st Qu.:0.14720   1st Qu.:0.1145   1st Qu.:0.06493     
    ##  Median :0.13130   Median :0.21190   Median :0.2267   Median :0.09993     
    ##  Mean   :0.13237   Mean   :0.25427   Mean   :0.2722   Mean   :0.11461     
    ##  3rd Qu.:0.14600   3rd Qu.:0.33910   3rd Qu.:0.3829   3rd Qu.:0.16140     
    ##  Max.   :0.22260   Max.   :1.05800   Max.   :1.2520   Max.   :0.29100     
    ##  symmetry_worst   fractal_dimension_worst
    ##  Min.   :0.1565   Min.   :0.05504        
    ##  1st Qu.:0.2504   1st Qu.:0.07146        
    ##  Median :0.2822   Median :0.08004        
    ##  Mean   :0.2901   Mean   :0.08395        
    ##  3rd Qu.:0.3179   3rd Qu.:0.09208        
    ##  Max.   :0.6638   Max.   :0.20750

``` r
# remove id

wbcd <- wbcd[-1]
```

Lets try a better visualization with visdat

``` r
vis_dat(wbcd)
```

![](kNN_Cancer_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
vis_miss(wbcd)
```

![](kNN_Cancer_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

I’d like to make one asser statement regarding the radius of the tumers

``` r
invisible(
  wbcd %>% 
    verify(c( "B", "M") %in% .$diagnosis) %>% 
          assert(within_bounds(0,Inf), radius_mean) 
)
```

Convert diagnosis into
factors

``` r
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("M", "B"), labels = c("Malignant", "Benign"))
round(prop.table(table(wbcd$diagnosis))*100, digits = 1)
```

    ## 
    ## Malignant    Benign 
    ##      37.3      62.7

Since different features have different scales, we introduce a function
“normalize” to set the values withing a range of \[0,1\]

``` r
normalize <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
  
  
}
```

use lapply to get a list (the l in lappy) and then convert the list inso
data frame

``` r
wbcd_n <- apply(wbcd[2:31],2,normalize)
#another choice 
#wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
```

We split the data into training and testing (the records wehre already
in random order, so no need to randomize further)

``` r
wbcd_train <- wbcd_n[1:469,]
wbcd_test <- wbcd_n[470:569,]
```

remove the labeling

``` r
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569,1]
```

Building the
classifier

``` r
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k = 21)
```

Evaluating model performance

``` r
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels | Malignant |    Benign | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |        21 |         2 |        23 | 
    ##                  |     0.913 |     0.087 |     0.230 | 
    ##                  |     1.000 |     0.025 |           | 
    ##                  |     0.210 |     0.020 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |         0 |        77 |        77 | 
    ##                  |     0.000 |     1.000 |     0.770 | 
    ##                  |     0.000 |     0.975 |           | 
    ##                  |     0.000 |     0.770 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        21 |        79 |       100 | 
    ##                  |     0.210 |     0.790 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ## 

Tuning See what caret has to say about knn

``` r
modelLookup("knn")
```

    ##   model parameter      label forReg forClass probModel
    ## 1   knn         k #Neighbors   TRUE     TRUE      TRUE

Let us do the tuning with some values for k

``` r
m <- train(wbcd_train,  wbcd_train_labels, method = "knn")
m
```

    ## k-Nearest Neighbors 
    ## 
    ## 469 samples
    ##  30 predictor
    ##   2 classes: 'Malignant', 'Benign' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 469, 469, 469, 469, 469, 469, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k  Accuracy   Kappa    
    ##   5  0.9567915  0.9094579
    ##   7  0.9586801  0.9132557
    ##   9  0.9603188  0.9167248
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 9.

Something
fancier

``` r
m_cv <- train(wbcd_train, wbcd_train_labels, method = "knn",  trControl = trainControl(method = "cv"), tuneLength = 15) 
# here tuneLength is how many values of k we are going to use
m_boot <- train(wbcd_train, wbcd_train_labels, method = "knn",  trControl = trainControl(method = "boot"), tuneLength = 15)

ctrl <- trainControl(classProbs = TRUE, method = "boot", verboseIter = TRUE)
m_boot_ROC <- train(wbcd_train, wbcd_train_labels, method = "knn",  trControl = ctrl, tuneLength = 15, metric = "ROC")
```

    ## Warning in train.default(wbcd_train, wbcd_train_labels, method = "knn", :
    ## The metric "ROC" was not in the result set. Accuracy will be used instead.

    ## + Resample01: k= 5 
    ## - Resample01: k= 5 
    ## + Resample01: k= 7 
    ## - Resample01: k= 7 
    ## + Resample01: k= 9 
    ## - Resample01: k= 9 
    ## + Resample01: k=11 
    ## - Resample01: k=11 
    ## + Resample01: k=13 
    ## - Resample01: k=13 
    ## + Resample01: k=15 
    ## - Resample01: k=15 
    ## + Resample01: k=17 
    ## - Resample01: k=17 
    ## + Resample01: k=19 
    ## - Resample01: k=19 
    ## + Resample01: k=21 
    ## - Resample01: k=21 
    ## + Resample01: k=23 
    ## - Resample01: k=23 
    ## + Resample01: k=25 
    ## - Resample01: k=25 
    ## + Resample01: k=27 
    ## - Resample01: k=27 
    ## + Resample01: k=29 
    ## - Resample01: k=29 
    ## + Resample01: k=31 
    ## - Resample01: k=31 
    ## + Resample01: k=33 
    ## - Resample01: k=33 
    ## + Resample02: k= 5 
    ## - Resample02: k= 5 
    ## + Resample02: k= 7 
    ## - Resample02: k= 7 
    ## + Resample02: k= 9 
    ## - Resample02: k= 9 
    ## + Resample02: k=11 
    ## - Resample02: k=11 
    ## + Resample02: k=13 
    ## - Resample02: k=13 
    ## + Resample02: k=15 
    ## - Resample02: k=15 
    ## + Resample02: k=17 
    ## - Resample02: k=17 
    ## + Resample02: k=19 
    ## - Resample02: k=19 
    ## + Resample02: k=21 
    ## - Resample02: k=21 
    ## + Resample02: k=23 
    ## - Resample02: k=23 
    ## + Resample02: k=25 
    ## - Resample02: k=25 
    ## + Resample02: k=27 
    ## - Resample02: k=27 
    ## + Resample02: k=29 
    ## - Resample02: k=29 
    ## + Resample02: k=31 
    ## - Resample02: k=31 
    ## + Resample02: k=33 
    ## - Resample02: k=33 
    ## + Resample03: k= 5 
    ## - Resample03: k= 5 
    ## + Resample03: k= 7 
    ## - Resample03: k= 7 
    ## + Resample03: k= 9 
    ## - Resample03: k= 9 
    ## + Resample03: k=11 
    ## - Resample03: k=11 
    ## + Resample03: k=13 
    ## - Resample03: k=13 
    ## + Resample03: k=15 
    ## - Resample03: k=15 
    ## + Resample03: k=17 
    ## - Resample03: k=17 
    ## + Resample03: k=19 
    ## - Resample03: k=19 
    ## + Resample03: k=21 
    ## - Resample03: k=21 
    ## + Resample03: k=23 
    ## - Resample03: k=23 
    ## + Resample03: k=25 
    ## - Resample03: k=25 
    ## + Resample03: k=27 
    ## - Resample03: k=27 
    ## + Resample03: k=29 
    ## - Resample03: k=29 
    ## + Resample03: k=31 
    ## - Resample03: k=31 
    ## + Resample03: k=33 
    ## - Resample03: k=33 
    ## + Resample04: k= 5 
    ## - Resample04: k= 5 
    ## + Resample04: k= 7 
    ## - Resample04: k= 7 
    ## + Resample04: k= 9 
    ## - Resample04: k= 9 
    ## + Resample04: k=11 
    ## - Resample04: k=11 
    ## + Resample04: k=13 
    ## - Resample04: k=13 
    ## + Resample04: k=15 
    ## - Resample04: k=15 
    ## + Resample04: k=17 
    ## - Resample04: k=17 
    ## + Resample04: k=19 
    ## - Resample04: k=19 
    ## + Resample04: k=21 
    ## - Resample04: k=21 
    ## + Resample04: k=23 
    ## - Resample04: k=23 
    ## + Resample04: k=25 
    ## - Resample04: k=25 
    ## + Resample04: k=27 
    ## - Resample04: k=27 
    ## + Resample04: k=29 
    ## - Resample04: k=29 
    ## + Resample04: k=31 
    ## - Resample04: k=31 
    ## + Resample04: k=33 
    ## - Resample04: k=33 
    ## + Resample05: k= 5 
    ## - Resample05: k= 5 
    ## + Resample05: k= 7 
    ## - Resample05: k= 7 
    ## + Resample05: k= 9 
    ## - Resample05: k= 9 
    ## + Resample05: k=11 
    ## - Resample05: k=11 
    ## + Resample05: k=13 
    ## - Resample05: k=13 
    ## + Resample05: k=15 
    ## - Resample05: k=15 
    ## + Resample05: k=17 
    ## - Resample05: k=17 
    ## + Resample05: k=19 
    ## - Resample05: k=19 
    ## + Resample05: k=21 
    ## - Resample05: k=21 
    ## + Resample05: k=23 
    ## - Resample05: k=23 
    ## + Resample05: k=25 
    ## - Resample05: k=25 
    ## + Resample05: k=27 
    ## - Resample05: k=27 
    ## + Resample05: k=29 
    ## - Resample05: k=29 
    ## + Resample05: k=31 
    ## - Resample05: k=31 
    ## + Resample05: k=33 
    ## - Resample05: k=33 
    ## + Resample06: k= 5 
    ## - Resample06: k= 5 
    ## + Resample06: k= 7 
    ## - Resample06: k= 7 
    ## + Resample06: k= 9 
    ## - Resample06: k= 9 
    ## + Resample06: k=11 
    ## - Resample06: k=11 
    ## + Resample06: k=13 
    ## - Resample06: k=13 
    ## + Resample06: k=15 
    ## - Resample06: k=15 
    ## + Resample06: k=17 
    ## - Resample06: k=17 
    ## + Resample06: k=19 
    ## - Resample06: k=19 
    ## + Resample06: k=21 
    ## - Resample06: k=21 
    ## + Resample06: k=23 
    ## - Resample06: k=23 
    ## + Resample06: k=25 
    ## - Resample06: k=25 
    ## + Resample06: k=27 
    ## - Resample06: k=27 
    ## + Resample06: k=29 
    ## - Resample06: k=29 
    ## + Resample06: k=31 
    ## - Resample06: k=31 
    ## + Resample06: k=33 
    ## - Resample06: k=33 
    ## + Resample07: k= 5 
    ## - Resample07: k= 5 
    ## + Resample07: k= 7 
    ## - Resample07: k= 7 
    ## + Resample07: k= 9 
    ## - Resample07: k= 9 
    ## + Resample07: k=11 
    ## - Resample07: k=11 
    ## + Resample07: k=13 
    ## - Resample07: k=13 
    ## + Resample07: k=15 
    ## - Resample07: k=15 
    ## + Resample07: k=17 
    ## - Resample07: k=17 
    ## + Resample07: k=19 
    ## - Resample07: k=19 
    ## + Resample07: k=21 
    ## - Resample07: k=21 
    ## + Resample07: k=23 
    ## - Resample07: k=23 
    ## + Resample07: k=25 
    ## - Resample07: k=25 
    ## + Resample07: k=27 
    ## - Resample07: k=27 
    ## + Resample07: k=29 
    ## - Resample07: k=29 
    ## + Resample07: k=31 
    ## - Resample07: k=31 
    ## + Resample07: k=33 
    ## - Resample07: k=33 
    ## + Resample08: k= 5 
    ## - Resample08: k= 5 
    ## + Resample08: k= 7 
    ## - Resample08: k= 7 
    ## + Resample08: k= 9 
    ## - Resample08: k= 9 
    ## + Resample08: k=11 
    ## - Resample08: k=11 
    ## + Resample08: k=13 
    ## - Resample08: k=13 
    ## + Resample08: k=15 
    ## - Resample08: k=15 
    ## + Resample08: k=17 
    ## - Resample08: k=17 
    ## + Resample08: k=19 
    ## - Resample08: k=19 
    ## + Resample08: k=21 
    ## - Resample08: k=21 
    ## + Resample08: k=23 
    ## - Resample08: k=23 
    ## + Resample08: k=25 
    ## - Resample08: k=25 
    ## + Resample08: k=27 
    ## - Resample08: k=27 
    ## + Resample08: k=29 
    ## - Resample08: k=29 
    ## + Resample08: k=31 
    ## - Resample08: k=31 
    ## + Resample08: k=33 
    ## - Resample08: k=33 
    ## + Resample09: k= 5 
    ## - Resample09: k= 5 
    ## + Resample09: k= 7 
    ## - Resample09: k= 7 
    ## + Resample09: k= 9 
    ## - Resample09: k= 9 
    ## + Resample09: k=11 
    ## - Resample09: k=11 
    ## + Resample09: k=13 
    ## - Resample09: k=13 
    ## + Resample09: k=15 
    ## - Resample09: k=15 
    ## + Resample09: k=17 
    ## - Resample09: k=17 
    ## + Resample09: k=19 
    ## - Resample09: k=19 
    ## + Resample09: k=21 
    ## - Resample09: k=21 
    ## + Resample09: k=23 
    ## - Resample09: k=23 
    ## + Resample09: k=25 
    ## - Resample09: k=25 
    ## + Resample09: k=27 
    ## - Resample09: k=27 
    ## + Resample09: k=29 
    ## - Resample09: k=29 
    ## + Resample09: k=31 
    ## - Resample09: k=31 
    ## + Resample09: k=33 
    ## - Resample09: k=33 
    ## + Resample10: k= 5 
    ## - Resample10: k= 5 
    ## + Resample10: k= 7 
    ## - Resample10: k= 7 
    ## + Resample10: k= 9 
    ## - Resample10: k= 9 
    ## + Resample10: k=11 
    ## - Resample10: k=11 
    ## + Resample10: k=13 
    ## - Resample10: k=13 
    ## + Resample10: k=15 
    ## - Resample10: k=15 
    ## + Resample10: k=17 
    ## - Resample10: k=17 
    ## + Resample10: k=19 
    ## - Resample10: k=19 
    ## + Resample10: k=21 
    ## - Resample10: k=21 
    ## + Resample10: k=23 
    ## - Resample10: k=23 
    ## + Resample10: k=25 
    ## - Resample10: k=25 
    ## + Resample10: k=27 
    ## - Resample10: k=27 
    ## + Resample10: k=29 
    ## - Resample10: k=29 
    ## + Resample10: k=31 
    ## - Resample10: k=31 
    ## + Resample10: k=33 
    ## - Resample10: k=33 
    ## + Resample11: k= 5 
    ## - Resample11: k= 5 
    ## + Resample11: k= 7 
    ## - Resample11: k= 7 
    ## + Resample11: k= 9 
    ## - Resample11: k= 9 
    ## + Resample11: k=11 
    ## - Resample11: k=11 
    ## + Resample11: k=13 
    ## - Resample11: k=13 
    ## + Resample11: k=15 
    ## - Resample11: k=15 
    ## + Resample11: k=17 
    ## - Resample11: k=17 
    ## + Resample11: k=19 
    ## - Resample11: k=19 
    ## + Resample11: k=21 
    ## - Resample11: k=21 
    ## + Resample11: k=23 
    ## - Resample11: k=23 
    ## + Resample11: k=25 
    ## - Resample11: k=25 
    ## + Resample11: k=27 
    ## - Resample11: k=27 
    ## + Resample11: k=29 
    ## - Resample11: k=29 
    ## + Resample11: k=31 
    ## - Resample11: k=31 
    ## + Resample11: k=33 
    ## - Resample11: k=33 
    ## + Resample12: k= 5 
    ## - Resample12: k= 5 
    ## + Resample12: k= 7 
    ## - Resample12: k= 7 
    ## + Resample12: k= 9 
    ## - Resample12: k= 9 
    ## + Resample12: k=11 
    ## - Resample12: k=11 
    ## + Resample12: k=13 
    ## - Resample12: k=13 
    ## + Resample12: k=15 
    ## - Resample12: k=15 
    ## + Resample12: k=17 
    ## - Resample12: k=17 
    ## + Resample12: k=19 
    ## - Resample12: k=19 
    ## + Resample12: k=21 
    ## - Resample12: k=21 
    ## + Resample12: k=23 
    ## - Resample12: k=23 
    ## + Resample12: k=25 
    ## - Resample12: k=25 
    ## + Resample12: k=27 
    ## - Resample12: k=27 
    ## + Resample12: k=29 
    ## - Resample12: k=29 
    ## + Resample12: k=31 
    ## - Resample12: k=31 
    ## + Resample12: k=33 
    ## - Resample12: k=33 
    ## + Resample13: k= 5 
    ## - Resample13: k= 5 
    ## + Resample13: k= 7 
    ## - Resample13: k= 7 
    ## + Resample13: k= 9 
    ## - Resample13: k= 9 
    ## + Resample13: k=11 
    ## - Resample13: k=11 
    ## + Resample13: k=13 
    ## - Resample13: k=13 
    ## + Resample13: k=15 
    ## - Resample13: k=15 
    ## + Resample13: k=17 
    ## - Resample13: k=17 
    ## + Resample13: k=19 
    ## - Resample13: k=19 
    ## + Resample13: k=21 
    ## - Resample13: k=21 
    ## + Resample13: k=23 
    ## - Resample13: k=23 
    ## + Resample13: k=25 
    ## - Resample13: k=25 
    ## + Resample13: k=27 
    ## - Resample13: k=27 
    ## + Resample13: k=29 
    ## - Resample13: k=29 
    ## + Resample13: k=31 
    ## - Resample13: k=31 
    ## + Resample13: k=33 
    ## - Resample13: k=33 
    ## + Resample14: k= 5 
    ## - Resample14: k= 5 
    ## + Resample14: k= 7 
    ## - Resample14: k= 7 
    ## + Resample14: k= 9 
    ## - Resample14: k= 9 
    ## + Resample14: k=11 
    ## - Resample14: k=11 
    ## + Resample14: k=13 
    ## - Resample14: k=13 
    ## + Resample14: k=15 
    ## - Resample14: k=15 
    ## + Resample14: k=17 
    ## - Resample14: k=17 
    ## + Resample14: k=19 
    ## - Resample14: k=19 
    ## + Resample14: k=21 
    ## - Resample14: k=21 
    ## + Resample14: k=23 
    ## - Resample14: k=23 
    ## + Resample14: k=25 
    ## - Resample14: k=25 
    ## + Resample14: k=27 
    ## - Resample14: k=27 
    ## + Resample14: k=29 
    ## - Resample14: k=29 
    ## + Resample14: k=31 
    ## - Resample14: k=31 
    ## + Resample14: k=33 
    ## - Resample14: k=33 
    ## + Resample15: k= 5 
    ## - Resample15: k= 5 
    ## + Resample15: k= 7 
    ## - Resample15: k= 7 
    ## + Resample15: k= 9 
    ## - Resample15: k= 9 
    ## + Resample15: k=11 
    ## - Resample15: k=11 
    ## + Resample15: k=13 
    ## - Resample15: k=13 
    ## + Resample15: k=15 
    ## - Resample15: k=15 
    ## + Resample15: k=17 
    ## - Resample15: k=17 
    ## + Resample15: k=19 
    ## - Resample15: k=19 
    ## + Resample15: k=21 
    ## - Resample15: k=21 
    ## + Resample15: k=23 
    ## - Resample15: k=23 
    ## + Resample15: k=25 
    ## - Resample15: k=25 
    ## + Resample15: k=27 
    ## - Resample15: k=27 
    ## + Resample15: k=29 
    ## - Resample15: k=29 
    ## + Resample15: k=31 
    ## - Resample15: k=31 
    ## + Resample15: k=33 
    ## - Resample15: k=33 
    ## + Resample16: k= 5 
    ## - Resample16: k= 5 
    ## + Resample16: k= 7 
    ## - Resample16: k= 7 
    ## + Resample16: k= 9 
    ## - Resample16: k= 9 
    ## + Resample16: k=11 
    ## - Resample16: k=11 
    ## + Resample16: k=13 
    ## - Resample16: k=13 
    ## + Resample16: k=15 
    ## - Resample16: k=15 
    ## + Resample16: k=17 
    ## - Resample16: k=17 
    ## + Resample16: k=19 
    ## - Resample16: k=19 
    ## + Resample16: k=21 
    ## - Resample16: k=21 
    ## + Resample16: k=23 
    ## - Resample16: k=23 
    ## + Resample16: k=25 
    ## - Resample16: k=25 
    ## + Resample16: k=27 
    ## - Resample16: k=27 
    ## + Resample16: k=29 
    ## - Resample16: k=29 
    ## + Resample16: k=31 
    ## - Resample16: k=31 
    ## + Resample16: k=33 
    ## - Resample16: k=33 
    ## + Resample17: k= 5 
    ## - Resample17: k= 5 
    ## + Resample17: k= 7 
    ## - Resample17: k= 7 
    ## + Resample17: k= 9 
    ## - Resample17: k= 9 
    ## + Resample17: k=11 
    ## - Resample17: k=11 
    ## + Resample17: k=13 
    ## - Resample17: k=13 
    ## + Resample17: k=15 
    ## - Resample17: k=15 
    ## + Resample17: k=17 
    ## - Resample17: k=17 
    ## + Resample17: k=19 
    ## - Resample17: k=19 
    ## + Resample17: k=21 
    ## - Resample17: k=21 
    ## + Resample17: k=23 
    ## - Resample17: k=23 
    ## + Resample17: k=25 
    ## - Resample17: k=25 
    ## + Resample17: k=27 
    ## - Resample17: k=27 
    ## + Resample17: k=29 
    ## - Resample17: k=29 
    ## + Resample17: k=31 
    ## - Resample17: k=31 
    ## + Resample17: k=33 
    ## - Resample17: k=33 
    ## + Resample18: k= 5 
    ## - Resample18: k= 5 
    ## + Resample18: k= 7 
    ## - Resample18: k= 7 
    ## + Resample18: k= 9 
    ## - Resample18: k= 9 
    ## + Resample18: k=11 
    ## - Resample18: k=11 
    ## + Resample18: k=13 
    ## - Resample18: k=13 
    ## + Resample18: k=15 
    ## - Resample18: k=15 
    ## + Resample18: k=17 
    ## - Resample18: k=17 
    ## + Resample18: k=19 
    ## - Resample18: k=19 
    ## + Resample18: k=21 
    ## - Resample18: k=21 
    ## + Resample18: k=23 
    ## - Resample18: k=23 
    ## + Resample18: k=25 
    ## - Resample18: k=25 
    ## + Resample18: k=27 
    ## - Resample18: k=27 
    ## + Resample18: k=29 
    ## - Resample18: k=29 
    ## + Resample18: k=31 
    ## - Resample18: k=31 
    ## + Resample18: k=33 
    ## - Resample18: k=33 
    ## + Resample19: k= 5 
    ## - Resample19: k= 5 
    ## + Resample19: k= 7 
    ## - Resample19: k= 7 
    ## + Resample19: k= 9 
    ## - Resample19: k= 9 
    ## + Resample19: k=11 
    ## - Resample19: k=11 
    ## + Resample19: k=13 
    ## - Resample19: k=13 
    ## + Resample19: k=15 
    ## - Resample19: k=15 
    ## + Resample19: k=17 
    ## - Resample19: k=17 
    ## + Resample19: k=19 
    ## - Resample19: k=19 
    ## + Resample19: k=21 
    ## - Resample19: k=21 
    ## + Resample19: k=23 
    ## - Resample19: k=23 
    ## + Resample19: k=25 
    ## - Resample19: k=25 
    ## + Resample19: k=27 
    ## - Resample19: k=27 
    ## + Resample19: k=29 
    ## - Resample19: k=29 
    ## + Resample19: k=31 
    ## - Resample19: k=31 
    ## + Resample19: k=33 
    ## - Resample19: k=33 
    ## + Resample20: k= 5 
    ## - Resample20: k= 5 
    ## + Resample20: k= 7 
    ## - Resample20: k= 7 
    ## + Resample20: k= 9 
    ## - Resample20: k= 9 
    ## + Resample20: k=11 
    ## - Resample20: k=11 
    ## + Resample20: k=13 
    ## - Resample20: k=13 
    ## + Resample20: k=15 
    ## - Resample20: k=15 
    ## + Resample20: k=17 
    ## - Resample20: k=17 
    ## + Resample20: k=19 
    ## - Resample20: k=19 
    ## + Resample20: k=21 
    ## - Resample20: k=21 
    ## + Resample20: k=23 
    ## - Resample20: k=23 
    ## + Resample20: k=25 
    ## - Resample20: k=25 
    ## + Resample20: k=27 
    ## - Resample20: k=27 
    ## + Resample20: k=29 
    ## - Resample20: k=29 
    ## + Resample20: k=31 
    ## - Resample20: k=31 
    ## + Resample20: k=33 
    ## - Resample20: k=33 
    ## + Resample21: k= 5 
    ## - Resample21: k= 5 
    ## + Resample21: k= 7 
    ## - Resample21: k= 7 
    ## + Resample21: k= 9 
    ## - Resample21: k= 9 
    ## + Resample21: k=11 
    ## - Resample21: k=11 
    ## + Resample21: k=13 
    ## - Resample21: k=13 
    ## + Resample21: k=15 
    ## - Resample21: k=15 
    ## + Resample21: k=17 
    ## - Resample21: k=17 
    ## + Resample21: k=19 
    ## - Resample21: k=19 
    ## + Resample21: k=21 
    ## - Resample21: k=21 
    ## + Resample21: k=23 
    ## - Resample21: k=23 
    ## + Resample21: k=25 
    ## - Resample21: k=25 
    ## + Resample21: k=27 
    ## - Resample21: k=27 
    ## + Resample21: k=29 
    ## - Resample21: k=29 
    ## + Resample21: k=31 
    ## - Resample21: k=31 
    ## + Resample21: k=33 
    ## - Resample21: k=33 
    ## + Resample22: k= 5 
    ## - Resample22: k= 5 
    ## + Resample22: k= 7 
    ## - Resample22: k= 7 
    ## + Resample22: k= 9 
    ## - Resample22: k= 9 
    ## + Resample22: k=11 
    ## - Resample22: k=11 
    ## + Resample22: k=13 
    ## - Resample22: k=13 
    ## + Resample22: k=15 
    ## - Resample22: k=15 
    ## + Resample22: k=17 
    ## - Resample22: k=17 
    ## + Resample22: k=19 
    ## - Resample22: k=19 
    ## + Resample22: k=21 
    ## - Resample22: k=21 
    ## + Resample22: k=23 
    ## - Resample22: k=23 
    ## + Resample22: k=25 
    ## - Resample22: k=25 
    ## + Resample22: k=27 
    ## - Resample22: k=27 
    ## + Resample22: k=29 
    ## - Resample22: k=29 
    ## + Resample22: k=31 
    ## - Resample22: k=31 
    ## + Resample22: k=33 
    ## - Resample22: k=33 
    ## + Resample23: k= 5 
    ## - Resample23: k= 5 
    ## + Resample23: k= 7 
    ## - Resample23: k= 7 
    ## + Resample23: k= 9 
    ## - Resample23: k= 9 
    ## + Resample23: k=11 
    ## - Resample23: k=11 
    ## + Resample23: k=13 
    ## - Resample23: k=13 
    ## + Resample23: k=15 
    ## - Resample23: k=15 
    ## + Resample23: k=17 
    ## - Resample23: k=17 
    ## + Resample23: k=19 
    ## - Resample23: k=19 
    ## + Resample23: k=21 
    ## - Resample23: k=21 
    ## + Resample23: k=23 
    ## - Resample23: k=23 
    ## + Resample23: k=25 
    ## - Resample23: k=25 
    ## + Resample23: k=27 
    ## - Resample23: k=27 
    ## + Resample23: k=29 
    ## - Resample23: k=29 
    ## + Resample23: k=31 
    ## - Resample23: k=31 
    ## + Resample23: k=33 
    ## - Resample23: k=33 
    ## + Resample24: k= 5 
    ## - Resample24: k= 5 
    ## + Resample24: k= 7 
    ## - Resample24: k= 7 
    ## + Resample24: k= 9 
    ## - Resample24: k= 9 
    ## + Resample24: k=11 
    ## - Resample24: k=11 
    ## + Resample24: k=13 
    ## - Resample24: k=13 
    ## + Resample24: k=15 
    ## - Resample24: k=15 
    ## + Resample24: k=17 
    ## - Resample24: k=17 
    ## + Resample24: k=19 
    ## - Resample24: k=19 
    ## + Resample24: k=21 
    ## - Resample24: k=21 
    ## + Resample24: k=23 
    ## - Resample24: k=23 
    ## + Resample24: k=25 
    ## - Resample24: k=25 
    ## + Resample24: k=27 
    ## - Resample24: k=27 
    ## + Resample24: k=29 
    ## - Resample24: k=29 
    ## + Resample24: k=31 
    ## - Resample24: k=31 
    ## + Resample24: k=33 
    ## - Resample24: k=33 
    ## + Resample25: k= 5 
    ## - Resample25: k= 5 
    ## + Resample25: k= 7 
    ## - Resample25: k= 7 
    ## + Resample25: k= 9 
    ## - Resample25: k= 9 
    ## + Resample25: k=11 
    ## - Resample25: k=11 
    ## + Resample25: k=13 
    ## - Resample25: k=13 
    ## + Resample25: k=15 
    ## - Resample25: k=15 
    ## + Resample25: k=17 
    ## - Resample25: k=17 
    ## + Resample25: k=19 
    ## - Resample25: k=19 
    ## + Resample25: k=21 
    ## - Resample25: k=21 
    ## + Resample25: k=23 
    ## - Resample25: k=23 
    ## + Resample25: k=25 
    ## - Resample25: k=25 
    ## + Resample25: k=27 
    ## - Resample25: k=27 
    ## + Resample25: k=29 
    ## - Resample25: k=29 
    ## + Resample25: k=31 
    ## - Resample25: k=31 
    ## + Resample25: k=33 
    ## - Resample25: k=33 
    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting k = 11 on full training set

``` r
ctrl <- trainControl(method = "repeatedcv",   # 10fold cross validation
                     number = 5,                            # do 5 repititions of cv
                     summaryFunction = twoClassSummary, # Use AUC to pick the best model
                     classProbs = TRUE,
                     allowParallel = TRUE,
                     verboseIter = TRUE)
m_cv_ROC <- train(wbcd_train, wbcd_train_labels,
      method = "knn",
      metric = "ROC",
      trControl = ctrl)
```

    ## + Fold1.Rep1: k=5 
    ## - Fold1.Rep1: k=5 
    ## + Fold1.Rep1: k=7 
    ## - Fold1.Rep1: k=7 
    ## + Fold1.Rep1: k=9 
    ## - Fold1.Rep1: k=9 
    ## + Fold2.Rep1: k=5 
    ## - Fold2.Rep1: k=5 
    ## + Fold2.Rep1: k=7 
    ## - Fold2.Rep1: k=7 
    ## + Fold2.Rep1: k=9 
    ## - Fold2.Rep1: k=9 
    ## + Fold3.Rep1: k=5 
    ## - Fold3.Rep1: k=5 
    ## + Fold3.Rep1: k=7 
    ## - Fold3.Rep1: k=7 
    ## + Fold3.Rep1: k=9 
    ## - Fold3.Rep1: k=9 
    ## + Fold4.Rep1: k=5 
    ## - Fold4.Rep1: k=5 
    ## + Fold4.Rep1: k=7 
    ## - Fold4.Rep1: k=7 
    ## + Fold4.Rep1: k=9 
    ## - Fold4.Rep1: k=9 
    ## + Fold5.Rep1: k=5 
    ## - Fold5.Rep1: k=5 
    ## + Fold5.Rep1: k=7 
    ## - Fold5.Rep1: k=7 
    ## + Fold5.Rep1: k=9 
    ## - Fold5.Rep1: k=9 
    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting k = 9 on full training set

``` r
m_cv
```

    ## k-Nearest Neighbors 
    ## 
    ## 469 samples
    ##  30 predictor
    ##   2 classes: 'Malignant', 'Benign' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 422, 422, 422, 422, 422, 422, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k   Accuracy   Kappa    
    ##    5  0.9637835  0.9242593
    ##    7  0.9637835  0.9239493
    ##    9  0.9659112  0.9286337
    ##   11  0.9595282  0.9152727
    ##   13  0.9616559  0.9195769
    ##   15  0.9637835  0.9236466
    ##   17  0.9616559  0.9192669
    ##   19  0.9574006  0.9104324
    ##   21  0.9552729  0.9056620
    ##   23  0.9552729  0.9060540
    ##   25  0.9595282  0.9148120
    ##   27  0.9595282  0.9143458
    ##   29  0.9595282  0.9143458
    ##   31  0.9595282  0.9145059
    ##   33  0.9552729  0.9054413
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 9.

To get tons of details about the model and how it was tuned:

``` r
#str(m_cv)
```

Let us make a prediction

``` r
m_cv_ROC_prediction <- predict(m_cv_ROC,wbcd_test)
#CrossTable(x = wbcd_test_labels, y = m_cv_ROC_prediction, prop.chisq = FALSE)
confusionMatrix(m_cv_ROC_prediction, wbcd_test_labels)
```

    ## Confusion Matrix and Statistics
    ## 
    ##            Reference
    ## Prediction  Malignant Benign
    ##   Malignant        23      2
    ##   Benign            0     75
    ##                                           
    ##                Accuracy : 0.98            
    ##                  95% CI : (0.9296, 0.9976)
    ##     No Information Rate : 0.77            
    ##     P-Value [Acc > NIR] : 2.106e-09       
    ##                                           
    ##                   Kappa : 0.9452          
    ##  Mcnemar's Test P-Value : 0.4795          
    ##                                           
    ##             Sensitivity : 1.000           
    ##             Specificity : 0.974           
    ##          Pos Pred Value : 0.920           
    ##          Neg Pred Value : 1.000           
    ##              Prevalence : 0.230           
    ##          Detection Rate : 0.230           
    ##    Detection Prevalence : 0.250           
    ##       Balanced Accuracy : 0.987           
    ##                                           
    ##        'Positive' Class : Malignant       
    ## 

``` r
m_cv_ROC_prediction_probs <- predict(m_cv_ROC,wbcd_test, type = "prob") # you need the prob option to get ROC
#head(m_cv_ROC_prediction_probs)

ROC <- roc(predictor = m_cv_ROC_prediction_probs$Malignant,
               response = wbcd_test_labels,
               levels = rev(levels(wbcd_test_labels)))

ROC$auc
```

    ## Area under the curve: 0.9994

``` r
plot(ROC,main = "ROC for kNN")
```

![](kNN_Cancer_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->
