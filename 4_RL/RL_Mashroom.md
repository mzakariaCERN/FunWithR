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
library(RWeka) # for 1R learnerrs
## neeed to install java
## then 
#Sys.setenv(JAVA_HOME='C:\\Program Files/Java/jdk-10.0.1/')
#library(rJava) # for 1R learnerrs
```

Pulling the data: The mushroom data is from Brett Lantz's "Machine Learning with R" a repo for the data is under this [link](https://github.com/mzakariaCERN/Machine-Learning-with-R-datasets/blob/master/mushrooms.csv) and original data can be found in the [UCI Repository](https://archive.ics.uci.edu/ml)

``` r
mushrooms <- read.csv(file = "C:/Users/mkzak/Documents/GitHub/FunWithR/FunWithR/4_RL/Data/mushrooms.csv", stringsAsFactors = FALSE)
str(mushrooms)
```

    ## 'data.frame':    8124 obs. of  23 variables:
    ##  $ type                    : chr  "p" "e" "e" "p" ...
    ##  $ cap_shape               : chr  "x" "x" "b" "x" ...
    ##  $ cap_surface             : chr  "s" "s" "s" "y" ...
    ##  $ cap_color               : chr  "n" "y" "w" "w" ...
    ##  $ bruises                 : chr  "t" "t" "t" "t" ...
    ##  $ odor                    : chr  "p" "a" "l" "p" ...
    ##  $ gill_attachment         : chr  "f" "f" "f" "f" ...
    ##  $ gill_spacing            : chr  "c" "c" "c" "c" ...
    ##  $ gill_size               : chr  "n" "b" "b" "n" ...
    ##  $ gill_color              : chr  "k" "k" "n" "n" ...
    ##  $ stalk_shape             : chr  "e" "e" "e" "e" ...
    ##  $ stalk_root              : chr  "e" "c" "c" "e" ...
    ##  $ stalk_surface_above_ring: chr  "s" "s" "s" "s" ...
    ##  $ stalk_surface_below_ring: chr  "s" "s" "s" "s" ...
    ##  $ stalk_color_above_ring  : chr  "w" "w" "w" "w" ...
    ##  $ stalk_color_below_ring  : chr  "w" "w" "w" "w" ...
    ##  $ veil_type               : chr  "p" "p" "p" "p" ...
    ##  $ veil_color              : chr  "w" "w" "w" "w" ...
    ##  $ ring_number             : chr  "o" "o" "o" "o" ...
    ##  $ ring_type               : chr  "p" "p" "p" "p" ...
    ##  $ spore_print_color       : chr  "k" "n" "n" "k" ...
    ##  $ population              : chr  "s" "n" "n" "s" ...
    ##  $ habitat                 : chr  "u" "g" "m" "u" ...

``` r
summary(mushrooms)
```

    ##      type            cap_shape         cap_surface       
    ##  Length:8124        Length:8124        Length:8124       
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##   cap_color           bruises              odor          
    ##  Length:8124        Length:8124        Length:8124       
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##  gill_attachment    gill_spacing        gill_size        
    ##  Length:8124        Length:8124        Length:8124       
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##   gill_color        stalk_shape         stalk_root       
    ##  Length:8124        Length:8124        Length:8124       
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##  stalk_surface_above_ring stalk_surface_below_ring stalk_color_above_ring
    ##  Length:8124              Length:8124              Length:8124           
    ##  Class :character         Class :character         Class :character      
    ##  Mode  :character         Mode  :character         Mode  :character      
    ##  stalk_color_below_ring  veil_type          veil_color       
    ##  Length:8124            Length:8124        Length:8124       
    ##  Class :character       Class :character   Class :character  
    ##  Mode  :character       Mode  :character   Mode  :character  
    ##  ring_number         ring_type         spore_print_color 
    ##  Length:8124        Length:8124        Length:8124       
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##   population          habitat         
    ##  Length:8124        Length:8124       
    ##  Class :character   Class :character  
    ##  Mode  :character   Mode  :character

``` r
# mushrooms$veil_type has one one level, we might as well drop it
mushrooms$veil_type <- NULL
```

``` r
table(mushrooms$type)
```

    ## 
    ##    e    p 
    ## 4208 3916

Since we want to develop only rules for this sample (not some unforeseen mushroom type) we will not divide the data into testing and training.

If we use ZeroR classifier, it will classify all mushrooms as edible (because of the slight majority we saw in mushrooms$type). This is not very useful and we can do better using 1R learnerns.
