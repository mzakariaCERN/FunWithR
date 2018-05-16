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
library(RWeka) # for 1R and JRip learners
## neeed to install java
## see this video https://www.youtube.com/watch?v=Wp6uS7CmivE
## then 
#Sys.setenv(JAVA_HOME='C:\\Program Files/Java/jdk-10.0.1/')
#.rs.restartR()

library(rJava) # for 1R learnerrs
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

We notice two things to be fixed: 1. All data seems to be in string form, so we need to convert that to factors 2. veil\_type has only one level, so we better remove it

``` r
# just re read the data with setting the strings as factors
mushrooms <- read.csv(file = "C:/Users/mkzak/Documents/GitHub/FunWithR/FunWithR/4_RL/Data/mushrooms.csv")

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

``` r
mushroom_1R <- OneR(type ~ . , data = mushrooms)
mushroom_1R
```

    ## odor:
    ##  a   -> e
    ##  c   -> p
    ##  f   -> p
    ##  l   -> e
    ##  m   -> p
    ##  n   -> e
    ##  p   -> p
    ##  s   -> p
    ##  y   -> p
    ## (8004/8124 instances correct)

``` r
summary(mushroom_1R)
```

    ## 
    ## === Summary ===
    ## 
    ## Correctly Classified Instances        8004               98.5229 %
    ## Incorrectly Classified Instances       120                1.4771 %
    ## Kappa statistic                          0.9704
    ## Mean absolute error                      0.0148
    ## Root mean squared error                  0.1215
    ## Relative absolute error                  2.958  %
    ## Root relative squared error             24.323  %
    ## Total Number of Instances             8124     
    ## 
    ## === Confusion Matrix ===
    ## 
    ##     a    b   <-- classified as
    ##  4208    0 |    a = e
    ##   120 3796 |    b = p

Notice that the classifier accepted 120 poisonous mushrooms as edible!

``` r
mushroom_JRip <- JRip(type ~ ., data = mushrooms)
mushroom_JRip
```

    ## JRIP rules:
    ## ===========
    ## 
    ## (odor = f) => type=p (2160.0/0.0)
    ## (gill_size = n) and (gill_color = b) => type=p (1152.0/0.0)
    ## (gill_size = n) and (odor = p) => type=p (256.0/0.0)
    ## (odor = c) => type=p (192.0/0.0)
    ## (spore_print_color = r) => type=p (72.0/0.0)
    ## (stalk_surface_below_ring = y) and (stalk_surface_above_ring = k) => type=p (68.0/0.0)
    ## (habitat = l) and (cap_color = w) => type=p (8.0/0.0)
    ## (stalk_color_above_ring = y) => type=p (8.0/0.0)
    ##  => type=e (4208.0/0.0)
    ## 
    ## Number of Rules : 9

Notice that the model had no errors using these 9 rules. Also ntice that the rules go as if, else statements

``` r
summary(mushroom_JRip)
```

    ## 
    ## === Summary ===
    ## 
    ## Correctly Classified Instances        8124              100      %
    ## Incorrectly Classified Instances         0                0      %
    ## Kappa statistic                          1     
    ## Mean absolute error                      0     
    ## Root mean squared error                  0     
    ## Relative absolute error                  0      %
    ## Root relative squared error              0      %
    ## Total Number of Instances             8124     
    ## 
    ## === Confusion Matrix ===
    ## 
    ##     a    b   <-- classified as
    ##  4208    0 |    a = e
    ##     0 3916 |    b = p
