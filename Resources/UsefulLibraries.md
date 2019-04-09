Interesting Packages in R
================

Here I list interesting R packages to keep record

1.  todor It gives a nice TODO makrdown documen
    [link](https://github.com/dokato/todor)

2.  check.packeges Nice code how to check if packages are there, if not
    install them [link](https://gist.github.com/smithdanielle/9913897)  
    check.packages function: install and load multiple R packages.  
    Check to see if packages are installed. Install them if they are
    not, then load them into the R session.

<!-- end list -->

    check.packages <- function(pkg){
        new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
        if (length(new.pkg)) 
            install.packages(new.pkg, dependencies = TRUE)
        sapply(pkg, require, character.only = TRUE)
    }
    
    # Usage example
    packages<-c("ggplot2", "afex", "ez", "Hmisc", "pander", "plyr")
    check.packages(packages)

3.  visdat Makes a nice plot of the data frame, type of data in it, and
    if there are any missing values
    [link](https://cran.r-project.org/web/packages/visdat/vignettes/using_visdat.html)

4.  assertr The package includes functions designed to verify
    assumptions about data. Example: That the age is not negative\! or
    that rows are
    unique  
    [example](https://daranzolin.github.io/2018-01-19-preeda/)  
    [link](https://cran.r-project.org/web/packages/assertr/vignettes/assertr.html)

5.  skimr a frictionless approach to summary statistics that displays
    summary statistics the user can skim quickly to understand their
    data [link](https://ropensci.org/blog/2017/07/11/skimr/)
