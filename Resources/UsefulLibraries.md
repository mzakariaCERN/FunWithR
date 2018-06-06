Interesting Packages in R
================

Here I list interesting R packages to keep record

1.  todor It gives a nice TODO makrdown documen [link](https://github.com/dokato/todor)

2.  check.packeges Nice code how to check if packages are there, if not install them [link](https://gist.github.com/smithdanielle/9913897)
    \# check.packages function: install and load multiple R packages.
    \# Check to see if packages are installed. Install them if they are not, then load them into the R session.
    check.packages &lt;- function(pkg){ new.pkg &lt;- pkg\[!(pkg %in% installed.packages()\[, "Package"\])\] if (length(new.pkg)) install.packages(new.pkg, dependencies = TRUE) sapply(pkg, require, character.only = TRUE) }

Usage example
=============

packages&lt;-c("ggplot2", "afex", "ez", "Hmisc", "pander", "plyr") check.packages(packages)