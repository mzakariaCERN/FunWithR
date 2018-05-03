Bank Loan classifier with Naive Bays
================
Mohammed Zakaria

``` r
library(ggplot2)
#install.packages("C50") # for DT algorithm
library(C50)
```

Pulling the data: The cancer data is from Brett Lantz's "Machine Learning with R" a repo for the data is under this link: <https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/credit.csv> and original data can be found under <https://archive.ics.uci.edu/ml>

``` r
credit <- read.csv(file="C:/Users/mkzak/Documents/GitHub/FunWithR/FunWithR/3_DT/Data/credit.csv", stringsAsFactors = FALSE)
str(credit)
```

    ## 'data.frame':    1000 obs. of  21 variables:
    ##  $ checking_balance    : chr  "< 0 DM" "1 - 200 DM" "unknown" "< 0 DM" ...
    ##  $ months_loan_duration: int  6 48 12 42 24 36 24 36 12 30 ...
    ##  $ credit_history      : chr  "critical" "repaid" "critical" "repaid" ...
    ##  $ purpose             : chr  "radio/tv" "radio/tv" "education" "furniture" ...
    ##  $ amount              : int  1169 5951 2096 7882 4870 9055 2835 6948 3059 5234 ...
    ##  $ savings_balance     : chr  "unknown" "< 100 DM" "< 100 DM" "< 100 DM" ...
    ##  $ employment_length   : chr  "> 7 yrs" "1 - 4 yrs" "4 - 7 yrs" "4 - 7 yrs" ...
    ##  $ installment_rate    : int  4 2 2 2 3 2 3 2 2 4 ...
    ##  $ personal_status     : chr  "single male" "female" "single male" "single male" ...
    ##  $ other_debtors       : chr  "none" "none" "none" "guarantor" ...
    ##  $ residence_history   : int  4 2 3 4 4 4 4 2 4 2 ...
    ##  $ property            : chr  "real estate" "real estate" "real estate" "building society savings" ...
    ##  $ age                 : int  67 22 49 45 53 35 53 35 61 28 ...
    ##  $ installment_plan    : chr  "none" "none" "none" "none" ...
    ##  $ housing             : chr  "own" "own" "own" "for free" ...
    ##  $ existing_credits    : int  2 1 1 1 2 1 1 1 1 2 ...
    ##  $ job                 : chr  "skilled employee" "skilled employee" "unskilled resident" "skilled employee" ...
    ##  $ dependents          : int  1 1 2 2 2 2 1 1 1 1 ...
    ##  $ telephone           : chr  "yes" "none" "none" "none" ...
    ##  $ foreign_worker      : chr  "yes" "yes" "yes" "yes" ...
    ##  $ default             : int  1 2 1 1 2 1 1 1 1 2 ...

``` r
summary(credit)
```

    ##  checking_balance   months_loan_duration credit_history    
    ##  Length:1000        Min.   : 4.0         Length:1000       
    ##  Class :character   1st Qu.:12.0         Class :character  
    ##  Mode  :character   Median :18.0         Mode  :character  
    ##                     Mean   :20.9                           
    ##                     3rd Qu.:24.0                           
    ##                     Max.   :72.0                           
    ##    purpose              amount      savings_balance    employment_length 
    ##  Length:1000        Min.   :  250   Length:1000        Length:1000       
    ##  Class :character   1st Qu.: 1366   Class :character   Class :character  
    ##  Mode  :character   Median : 2320   Mode  :character   Mode  :character  
    ##                     Mean   : 3271                                        
    ##                     3rd Qu.: 3972                                        
    ##                     Max.   :18424                                        
    ##  installment_rate personal_status    other_debtors      residence_history
    ##  Min.   :1.000    Length:1000        Length:1000        Min.   :1.000    
    ##  1st Qu.:2.000    Class :character   Class :character   1st Qu.:2.000    
    ##  Median :3.000    Mode  :character   Mode  :character   Median :3.000    
    ##  Mean   :2.973                                          Mean   :2.845    
    ##  3rd Qu.:4.000                                          3rd Qu.:4.000    
    ##  Max.   :4.000                                          Max.   :4.000    
    ##    property              age        installment_plan     housing         
    ##  Length:1000        Min.   :19.00   Length:1000        Length:1000       
    ##  Class :character   1st Qu.:27.00   Class :character   Class :character  
    ##  Mode  :character   Median :33.00   Mode  :character   Mode  :character  
    ##                     Mean   :35.55                                        
    ##                     3rd Qu.:42.00                                        
    ##                     Max.   :75.00                                        
    ##  existing_credits     job              dependents     telephone        
    ##  Min.   :1.000    Length:1000        Min.   :1.000   Length:1000       
    ##  1st Qu.:1.000    Class :character   1st Qu.:1.000   Class :character  
    ##  Median :1.000    Mode  :character   Median :1.000   Mode  :character  
    ##  Mean   :1.407                       Mean   :1.155                     
    ##  3rd Qu.:2.000                       3rd Qu.:1.000                     
    ##  Max.   :4.000                       Max.   :2.000                     
    ##  foreign_worker        default   
    ##  Length:1000        Min.   :1.0  
    ##  Class :character   1st Qu.:1.0  
    ##  Mode  :character   Median :1.0  
    ##                     Mean   :1.3  
    ##                     3rd Qu.:2.0  
    ##                     Max.   :2.0

from str() we see that the target feature is actually numerical representing a categorical variable (default vs. no default)

``` r
credit$default <- factor(credit$default)
```

``` r
table(credit$checking_balance)
```

    ## 
    ##     < 0 DM   > 200 DM 1 - 200 DM    unknown 
    ##        274         63        269        394

``` r
table(credit$default)
```

    ## 
    ##   1   2 
    ## 700 300

We divide the data 90:10. WE cannot assume that the data is random. So let us do that

``` r
set.seed(123)
train_sample <- sample(1000, 900) # get 900 randomly selected numbers, each between 0 and 1000
str(train_sample)
```

    ##  int [1:900] 288 788 409 881 937 46 525 887 548 453 ...

``` r
credit_train <- credit[train_sample, ]
credit_test <- credit[-train_sample,]
```

``` r
prop.table(table(credit_train$default))
```

    ## 
    ##         1         2 
    ## 0.7033333 0.2966667

``` r
prop.table(table(credit_test$default))
```

    ## 
    ##    1    2 
    ## 0.67 0.33

Close! So we can proceed.

``` r
# remove the "default" feature since this is the target one
credit_model <- C5.0(credit_train[-21], credit_train$default)
credit_model
```

    ## 
    ## Call:
    ## C5.0.default(x = credit_train[-21], y = credit_train$default)
    ## 
    ## Classification Tree
    ## Number of samples: 900 
    ## Number of predictors: 20 
    ## 
    ## Tree size: 54 
    ## 
    ## Non-standard options: attempt to group attributes
