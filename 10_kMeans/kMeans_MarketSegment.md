Teen Market Segments using k Means
================
Mohammed Zakaria

This dataset was compiled by Brett Lantzz (Ch9, Machine Learning with
R)

``` r
teens <- read.csv(file = "C:/Users/mkzak/Documents/GitHub/FunWithR/FunWithR/10_kMeans/Data/snsdata.csv")
str(teens)
```

    ## 'data.frame':    30000 obs. of  40 variables:
    ##  $ gradyear    : int  2006 2006 2006 2006 2006 2006 2006 2006 2006 2006 ...
    ##  $ gender      : Factor w/ 2 levels "F","M": 2 1 2 1 NA 1 1 2 1 1 ...
    ##  $ age         : num  19 18.8 18.3 18.9 19 ...
    ##  $ friends     : int  7 0 69 0 10 142 72 17 52 39 ...
    ##  $ basketball  : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ football    : int  0 1 1 0 0 0 0 0 0 0 ...
    ##  $ soccer      : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ softball    : int  0 0 0 0 0 0 0 1 0 0 ...
    ##  $ volleyball  : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ swimming    : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ cheerleading: int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ baseball    : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ tennis      : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ sports      : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ cute        : int  0 1 0 1 0 0 0 0 0 1 ...
    ##  $ sex         : int  0 0 0 0 1 1 0 2 0 0 ...
    ##  $ sexy        : int  0 0 0 0 0 0 0 1 0 0 ...
    ##  $ hot         : int  0 0 0 0 0 0 0 0 0 1 ...
    ##  $ kissed      : int  0 0 0 0 5 0 0 0 0 0 ...
    ##  $ dance       : int  1 0 0 0 1 0 0 0 0 0 ...
    ##  $ band        : int  0 0 2 0 1 0 1 0 0 0 ...
    ##  $ marching    : int  0 0 0 0 0 1 1 0 0 0 ...
    ##  $ music       : int  0 2 1 0 3 2 0 1 0 1 ...
    ##  $ rock        : int  0 2 0 1 0 0 0 1 0 1 ...
    ##  $ god         : int  0 1 0 0 1 0 0 0 0 6 ...
    ##  $ church      : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ jesus       : int  0 0 0 0 0 0 0 0 0 2 ...
    ##  $ bible       : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ hair        : int  0 6 0 0 1 0 0 0 0 1 ...
    ##  $ dress       : int  0 4 0 0 0 1 0 0 0 0 ...
    ##  $ blonde      : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ mall        : int  0 1 0 0 0 0 2 0 0 0 ...
    ##  $ shopping    : int  0 0 0 0 2 1 0 0 0 1 ...
    ##  $ clothes     : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ hollister   : int  0 0 0 0 0 0 2 0 0 0 ...
    ##  $ abercrombie : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ die         : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ death       : int  0 0 1 0 0 0 0 0 0 0 ...
    ##  $ drunk       : int  0 0 0 0 1 1 0 0 0 0 ...
    ##  $ drugs       : int  0 0 0 0 1 0 0 0 0 0 ...

There is a problem with the gender features: we have NA

``` r
table(teens$gender) # doesn't show it!
```

    ## 
    ##     F     M 
    ## 22054  5222

``` r
table(teens$gender, useNA = "ifany")
```

    ## 
    ##     F     M  <NA> 
    ## 22054  5222  2724

``` r
summary(teens$gender)
```

    ##     F     M  NA's 
    ## 22054  5222  2724

``` r
summary(teens$age)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
    ##   3.086  16.312  17.287  17.994  18.259 106.927    5086

Also of a concern, anre the min and maximum values. This also needs some
cleaning up. A more reasonable range for the age for high school
students would be between 13 and 20 years old. Any age value outside of
this range should be treated as missing data as cannot trust it.

``` r
teens$age <- ifelse(teens$age >= 13 & teens$age < 20, teens$age, NA)
summary(teens$age)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
    ##   13.03   16.30   17.27   17.25   18.22   20.00    5523

One approach to handle themissing values with a categorical variable -
like age - is by assigning it to it’s own category. So we can have Male,
Female, and dummy code (unknown) as a third gender.

``` r
teens$female <- ifelse(teens$gender == "F" & !is.na(teens$gender), 1, 0)
teens$no_gender <- ifelse(is.na(teens$gender), 1, 0)
```

Check out the work

``` r
table(teens$gender, useNA = "ifany")
```

    ## 
    ##     F     M  <NA> 
    ## 22054  5222  2724

``` r
table(teens$female, useNA = "ifany")
```

    ## 
    ##     0     1 
    ##  7946 22054

``` r
table(teens$no_gender, useNA = "ifany")
```

    ## 
    ##     0     1 
    ## 27276  2724

For the 5k some values that are NA. we need to advise a different
strategy to infer a better age estimate. We can use graduation year to
obtain the value for age

``` r
# to do it for one year
mean(teens$age, na.rm = TRUE)
```

    ## [1] 17.25243

``` r
#to do it for the 4 years
aggregate(data = teens, age ~ gradyear, mean, na.rm = TRUE)
```

    ##   gradyear      age
    ## 1     2006 18.65586
    ## 2     2007 17.70617
    ## 3     2008 16.76770
    ## 4     2009 15.81957

For a nice discussion on using aggregate vs the apply family please
refer to:
<https://stackoverflow.com/questions/3505701/grouping-functions-tapply-by-aggregate-and-the-apply-family>

The fact that the outout of aggregate is a data frame can cause trouble,
so we define our own
function:

``` r
ave_age <- ave(teens$age, teens$gradyear, FUN = function(x) mean(x, na.rm = TRUE))
```

``` r
teens$age <- ifelse(is.na(teens$age), ave_age, teens$age)
```

For model building we will use kmeans from the stats package

``` r
library(stats)
```

The kmeans function requires all numeric features and they should be
standardized. Lets start with the 36 features represening interestss

``` r
interests <- teens[5:40]
```

``` r
interests_z <- as.data.frame(lapply(interests, scale))
```
