R Notebook
================

This notebook plays a bit with web scraping.

``` r
#install.packages("rvest")
library(rvest)
```

    ## Warning: package 'rvest' was built under R version 3.5.1

    ## Loading required package: xml2

``` r
packt_page <- read_html("https://www.packtpub.com")
packt_page
```

    ## {xml_document}
    ## <html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">
    ## [1] <head>\n<title>Packt Publishing | Technology Books, eBooks &amp; Vid ...
    ## [2] <body id="ppv4" class="with-logo">\n        <!-- Google Tag Manager  ...

If we want to scrape the papge title, we inspect the output of
packt\_page and we see there is only one title per page, wrapped withing

<title>

and \<\\title\> tages.

``` r
html_node(packt_page, "title")
```

    ## {xml_node}
    ## <title>

To convert this into plain text, we run the function below:

``` r
html_node(packt_page, "title") %>% html_text()
```

    ## [1] "Packt Publishing | Technology Books, eBooks & Videos"

Another example: Scrap a list of all R-packages on
CRAN

``` r
cran_ml <- read_html("http://cran.r-project.org/web/views/MachineLearning.html")
cran_ml
```

    ## {xml_document}
    ## <html xmlns="http://www.w3.org/1999/xhtml">
    ## [1] <head>\n<title>CRAN Task View: Machine Learning &amp; Statistical Le ...
    ## [2] <body>\n  <h2>CRAN Task View: Machine Learning &amp; Statistical Lea ...
