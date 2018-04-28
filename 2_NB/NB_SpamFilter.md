Spam SMS classifier with Naive Bays
================
Mohammed Zakaria

Required packages

``` r
#install.packages("class") # for kNN classification
#library(class)
#install.packages("gmodels") # for CrossTable function at the evaluation
library(gmodels)
#install.packages("caret") # for model tuning
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
#install.packages("e1071") # to give us Naive Bayes 
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

``` r
#install.packages("tm") # to handle text data
library(tm)
```

    ## Loading required package: NLP

    ## 
    ## Attaching package: 'NLP'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     annotate

``` r
#install.packages("SnowballC") # for steaming
library(SnowballC)
#install.packages("wordcloud2") # to create word cloud
library(wordcloud2)
```

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

After installing (loading) tm library, we need to create a container for all the text we are dealing with. This is called a corpus. WE will use VCorpus (for volatile corpus: corpus stored in memove, compare it to PCorpus, which is stored on disk).

``` r
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
typeof(sms_corpus)
```

    ## [1] "list"

corpus can read from pdf of MS word using readerControl parameter. Check it out! Notice also that corpus is a list objects. So we can manipulate it as such.

``` r
print(sms_corpus)
```

    ## <<VCorpus>>
    ## Metadata:  corpus specific: 0, document level (indexed): 0
    ## Content:  documents: 5574

``` r
inspect(sms_corpus[1:3])
```

    ## <<VCorpus>>
    ## Metadata:  corpus specific: 0, document level (indexed): 0
    ## Content:  documents: 3
    ## 
    ## [[1]]
    ## <<PlainTextDocument>>
    ## Metadata:  7
    ## Content:  chars: 111
    ## 
    ## [[2]]
    ## <<PlainTextDocument>>
    ## Metadata:  7
    ## Content:  chars: 29
    ## 
    ## [[3]]
    ## <<PlainTextDocument>>
    ## Metadata:  7
    ## Content:  chars: 155

To see one message, we need to grab that element in from the list, and conver it to characters

``` r
as.character(sms_corpus[[1]])
```

    ## [1] "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."

This can be generalized using lapply

``` r
lapply(sms_corpus[1:2], as.character)
```

    ## $`1`
    ## [1] "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
    ## 
    ## $`2`
    ## [1] "Ok lar... Joking wif u oni..."

Next, we need to clean the text from special characters, capital letters etc, and convert it into separate words.

First, convert to lower case

``` r
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
```

remember the first message starts with an upper case letter. Let us take a look now

``` r
as.character(sms_corpus_clean[[1]])
```

    ## [1] "go until jurong point, crazy.. available only in bugis n great world la e buffet... cine there got amore wat..."

next thing, is to remove numbers from SMS. Though some might be useful for the sender/receiver. It doesn't play much value in spam/ham classification.

``` r
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers) # no need for content_tranformer b/c removeNumbers is built in tm. to see other
# built in functions type getTransformations()
```

Next we remove filler (stop) words suc as: 'and', 'or', and 'but'

``` r
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())
```

Notice that stoprwords returns a vector of the words we need to consider as such. We can thus add a remove words from the list to modify the selection

Next, we remove any punctuation

``` r
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)
```

Notice that this function will join words that have punctuation marks between them without any space. Ex: "Hi...Hi" will become "HiHi" To fix this we can create a function like

``` r
replacePunctuation <- function(x){gsub("[[:punct:]]", " ", x)}
```

the above function can be used with tm\_map

Next, we return all words to their root (AKA: steaming). To do this install and/or load SnowballC

``` r
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
```

next, to remove any extra white space

``` r
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
```

Finally, we need to convert the messages into word (AKA tokenization). We will convert the corpus into Document Term Matrix: rows are SMS messages, and columns are word. Think of it as creating dummy variables for each word.

``` r
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

sms_dtm
```

    ## <<DocumentTermMatrix (documents: 5574, terms: 6604)>>
    ## Non-/sparse entries: 42631/36768065
    ## Sparsity           : 100%
    ## Maximal term length: 40
    ## Weighting          : term frequency (tf)

Notice that we could have create a DTM from the raw corpus with all the pre-processing in one command

``` r
sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(tolower = TRUE, removeNumbers = TRUE, stopwords = TRUE, removePunctuation = TRUE, steaming = TRUE))

sms_dtm2
```

    ## <<DocumentTermMatrix (documents: 5574, terms: 8364)>>
    ## Non-/sparse entries: 44221/46576715
    ## Sparsity           : 100%
    ## Maximal term length: 40
    ## Weighting          : term frequency (tf)

notice there are minor differences between the two matrices. this is mainly due to cleaning the corpus after spliting it into words in the second option. it also uses a slightly different choice for stop words. To fortce it to use the same stopwords:

``` r
stopWords = function(x){removeWords(x, stopwords())}
```

and replace stopwrods = TRUE with that function

Next step is splitting the data into training and testing. Since the data is randomized as is, no need to randomize it

``` r
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170: 5574, ]

sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170: 5574, ]$type
```

Checking that ratio of spam to ham is close in both samples

``` r
prop.table(table(sms_train_labels))
```

    ## sms_train_labels
    ##       ham      spam 
    ## 0.8647158 0.1352842

``` r
prop.table(table(sms_test_labels))
```

    ## sms_test_labels
    ##       ham      spam 
    ## 0.8697509 0.1302491

create a word cloud. Lantz uses package wordcount. But I Think it is depreciated. So I am using wordcount2. Which needs a data frame with word and frequency. which can be prepared from a TermDocumentMatrix (TDM, and not DTM)

``` r
myTdm <- as.matrix(TermDocumentMatrix(sms_corpus, control = list(tolower = TRUE, removeNumbers = TRUE, stopwords = TRUE, removePunctuation = TRUE, steaming = TRUE)))
FreqMat <- data.frame(ST = rownames(myTdm),
Freq = rowSums(myTdm),
row.names = NULL)


wc2<- wordcloud2(FreqMat, minSize = 50)
```

We can also do similar plots with ham and spam messages to compare between them

Since the TDM has one feature per word, we have over 6000 features. Many of these features for words that were mentioned once or twice. These are unlikely to be useful in the classification. so we set a frequency filter to remove any word mentioned less than 5 times.

``` r
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)
```

    ##  chr [1:1158] "â€“" "abiola" "abl" "abt" "accept" "access" "account" ...

remove every column that matches this vector

``` r
sms_dtm_freq_train <- sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test  <- sms_dtm_test[,  sms_freq_words]
```

NB classifiers typically use categorical features. This poses an issue for sparce matrices (DTM) since the cell are numeric and measure the number of times a word appears in the same massage. To change it to categorical we convert it to yes/no

``` r
convert_counts <- function(x){
  x <- ifelse(x > 0, "yes", "no")
}
```

we apply the function on the columns

``` r
sms_train <- apply(sms_dtm_freq_train, 2, convert_counts)
sms_test <-  apply(sms_dtm_freq_test, 2, convert_counts)
```

Add a new chunk by clicking the *Insert Chunk* button on the toolbar or by pressing *Ctrl+Alt+I*.

When you save the notebook, an HTML file containing the code and output will be saved alongside it (click the *Preview* button or press *Ctrl+Shift+K* to preview the HTML file).

The preview shows you a rendered HTML copy of the contents of the editor. Consequently, unlike *Knit*, *Preview* does not run any R code chunks. Instead, the output of the chunk when it was last run in the editor is displayed.
