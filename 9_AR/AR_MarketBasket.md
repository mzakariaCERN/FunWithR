AR\_MarketBastet
================
Mohammed Zakaria

Our market basket analysis will utilize the purchase data from one month of operation for a grocery store. We will see that we have about 9835 transaction during that period. or about 30-50 per hour (depending on how many hours it operates). This indicates a medium size store.

Getting the data

``` r
#install.packages("arules")
library(arules)
```

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'arules'

    ## The following objects are masked from 'package:base':
    ## 
    ##     abbreviate, write

``` r
# To see vignettes
#arules::arules
data("Groceries")
summary(Groceries)
```

    ## transactions as itemMatrix in sparse format with
    ##  9835 rows (elements/itemsets/transactions) and
    ##  169 columns (items) and a density of 0.02609146 
    ## 
    ## most frequent items:
    ##       whole milk other vegetables       rolls/buns             soda 
    ##             2513             1903             1809             1715 
    ##           yogurt          (Other) 
    ##             1372            34055 
    ## 
    ## element (itemset/transaction) length distribution:
    ## sizes
    ##    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15 
    ## 2159 1643 1299 1005  855  645  545  438  350  246  182  117   78   77   55 
    ##   16   17   18   19   20   21   22   23   24   26   27   28   29   32 
    ##   46   29   14   14    9   11    4    6    1    1    1    1    3    1 
    ## 
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   1.000   2.000   3.000   4.409   6.000  32.000 
    ## 
    ## includes extended item information - examples:
    ##        labels  level2           level1
    ## 1 frankfurter sausage meat and sausage
    ## 2     sausage sausage meat and sausage
    ## 3  liver loaf sausage meat and sausage

``` r
inspect(Groceries[1:10])
```

    ##      items                     
    ## [1]  {citrus fruit,            
    ##       semi-finished bread,     
    ##       margarine,               
    ##       ready soups}             
    ## [2]  {tropical fruit,          
    ##       yogurt,                  
    ##       coffee}                  
    ## [3]  {whole milk}              
    ## [4]  {pip fruit,               
    ##       yogurt,                  
    ##       cream cheese ,           
    ##       meat spreads}            
    ## [5]  {other vegetables,        
    ##       whole milk,              
    ##       condensed milk,          
    ##       long life bakery product}
    ## [6]  {whole milk,              
    ##       butter,                  
    ##       yogurt,                  
    ##       rice,                    
    ##       abrasive cleaner}        
    ## [7]  {rolls/buns}              
    ## [8]  {other vegetables,        
    ##       UHT-milk,                
    ##       rolls/buns,              
    ##       bottled beer,            
    ##       liquor (appetizer)}      
    ## [9]  {pot plants}              
    ## [10] {whole milk,              
    ##       cereals}

``` r
as(Groceries[1:10], "list")
```

    ## [[1]]
    ## [1] "citrus fruit"        "semi-finished bread" "margarine"          
    ## [4] "ready soups"        
    ## 
    ## [[2]]
    ## [1] "tropical fruit" "yogurt"         "coffee"        
    ## 
    ## [[3]]
    ## [1] "whole milk"
    ## 
    ## [[4]]
    ## [1] "pip fruit"     "yogurt"        "cream cheese " "meat spreads" 
    ## 
    ## [[5]]
    ## [1] "other vegetables"         "whole milk"              
    ## [3] "condensed milk"           "long life bakery product"
    ## 
    ## [[6]]
    ## [1] "whole milk"       "butter"           "yogurt"          
    ## [4] "rice"             "abrasive cleaner"
    ## 
    ## [[7]]
    ## [1] "rolls/buns"
    ## 
    ## [[8]]
    ## [1] "other vegetables"   "UHT-milk"           "rolls/buns"        
    ## [4] "bottled beer"       "liquor (appetizer)"
    ## 
    ## [[9]]
    ## [1] "pot plants"
    ## 
    ## [[10]]
    ## [1] "whole milk" "cereals"

Number of rows means how many transactions we have. Number of columns means how many different items we might have in each transaction. The density of 0.02609146 indicates the ratio of nonzero matrix cells to total number of elements in the matrix ( which is equal to columns\* rows = 1662115). Total number of items purchased = 1662115\* density = 43367 Average transaction contains 43367 / 9835 = 4.41 (which is the same value that summary gives us)

using inspect to see fea transaction records

``` r
inspect(Groceries[1:5])
```

    ##     items                     
    ## [1] {citrus fruit,            
    ##      semi-finished bread,     
    ##      margarine,               
    ##      ready soups}             
    ## [2] {tropical fruit,          
    ##      yogurt,                  
    ##      coffee}                  
    ## [3] {whole milk}              
    ## [4] {pip fruit,               
    ##      yogurt,                  
    ##      cream cheese ,           
    ##      meat spreads}            
    ## [5] {other vegetables,        
    ##      whole milk,              
    ##      condensed milk,          
    ##      long life bakery product}

if we want to see the specific columns of the data, you can use \[raw, column\] notation. We can combine this with itemFrequency() to see the proportions of the transactions that contain the item

``` r
inspect(Groceries[1:5, 1:3])
```

    ##     items
    ## [1] {}   
    ## [2] {}   
    ## [3] {}   
    ## [4] {}   
    ## [5] {}

``` r
itemFrequency(Groceries[, 1:4])
```

    ## frankfurter     sausage  liver loaf         ham 
    ## 0.058973055 0.093950178 0.005083884 0.026029487

Notice that the sparce matrix columns are sorted in alphabetical order.

To present these statistics visually, we use itemFrequencyPlot() function. Since the transaction data requires a large number of items, we will need to limit the ones appearing in the plot in order to produce a readable chart.

``` r
itemFrequencyPlot(Groceries, support = 0.1) # showing data with 10% support at least
```

![](AR_MarketBasket_files/figure-markdown_github/unnamed-chunk-5-1.png)

Alternatively, we can set the limit on the number of items we wish to see, ranked by frequency

``` r
itemFrequencyPlot(Groceries, topN = 20)
```

![](AR_MarketBasket_files/figure-markdown_github/unnamed-chunk-6-1.png)

WE can alse view the entire sparce matrix using functiion image

``` r
image(Groceries[1:5])
```

![](AR_MarketBasket_files/figure-markdown_github/unnamed-chunk-7-1.png)

Notice that we have 5 rosw (number of transactions we want to look at, and 169 columns, indicating each item in the shopping transactions) This plot can help us with some quality control 1- (Ex. if we found one item that repeats all the way, it can be a sign that the store might include it's name on the records by mistake) 2- Viewing data historically can show seasonal effects

Rather than showing a large data set, we can simply sample few events

``` r
image(sample(Groceries, 100))
```

![](AR_MarketBasket_files/figure-markdown_github/unnamed-chunk-8-1.png)

We will use the apriori algorithm in the arules package. WE might need to experiment few time with support and confidence levels to produce a reasonable number of association rules. Setting the level too high we will find no rules, or they would be too generic to be useful. If set too low, we will get a large nunmber of rulse, or worse, taking too much time to finish the learning task. Starting with the defauls settings (support = 0.1, confidence = 0.8). Since support level of 10% require the item to be visible at least 9385 \* 0.1 = 939 times, and that we have only 8 items such, we shouldn't expect many rules (if any)

``` r
apriori(Groceries)
```

    ## Apriori
    ## 
    ## Parameter specification:
    ##  confidence minval smax arem  aval originalSupport maxtime support minlen
    ##         0.8    0.1    1 none FALSE            TRUE       5     0.1      1
    ##  maxlen target   ext
    ##      10  rules FALSE
    ## 
    ## Algorithmic control:
    ##  filter tree heap memopt load sort verbose
    ##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
    ## 
    ## Absolute minimum support count: 983 
    ## 
    ## set item appearances ...[0 item(s)] done [0.00s].
    ## set transactions ...[169 item(s), 9835 transaction(s)] done [0.00s].
    ## sorting and recoding items ... [8 item(s)] done [0.00s].
    ## creating transaction tree ... done [0.00s].
    ## checking subsets of size 1 2 done [0.00s].
    ## writing ... [0 rule(s)] done [0.00s].
    ## creating S4 object  ... done [0.00s].

    ## set of 0 rules
