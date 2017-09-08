
library('rvest')
## below we set the link to the game page (notice the '')
url <- 'http://www.nhl.com/scores/htmlreports/20162017/ES030411.HTM'
#url <- 'http://www.nhl.com/scores/htmlreports/20162017/ES030133.HTM'
webpage <- read_html(url)
webpage
Full_data <-  html_nodes(webpage,'.rborder')
Essential <- html_nodes(webpage, '.tborder')
DateAccurate <- html_nodes(webpage, 'td')
Full_data_txt <- html_text(Full_data)
Full_Essential_txt <- html_text(Essential)
Full_DateAccurae_txt <- html_text(DateAccurate)

Full_Essential_txt[4] ## the string containing date
RefDate <- regmatches(Full_Essential_txt[4],regexpr("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]", Full_Essential_txt[4])) 
RefYear <- substring(RefDate, 1, 4)
RefMonth <- substring(RefDate, 6, 7)
RefDay <- substring(RefDate, 9, 11)

Full_DateAccurae_txt[14] ## the string containing the date
RefDate <- regmatches(Full_DateAccurae_txt[14], regexpr("[0-9][0-9], [0-9][0-9][0-9][0-9]", Full_DateAccurae_txt[14]))
RefYear <- substring(RefDate, 5, 8)
RefDay <- substring(RefDate, 1,2)
ToGetMonth <- regmatches(Full_DateAccurae_txt[14], regexpr(", [A-Z][a-z]+", Full_DateAccurae_txt[14]))
RefMonth_2 <- regmatches(ToGetMonth, regexpr("[A-Z][a-z]+", ToGetMonth))


RefDateProper <- paste(RefMonth_2, "/", RefDay, "/", RefYear)
testing <- as.data.frame(Full_data_txt)[27:525,1]
## here we set the patch for the csv files to be saved (notice "")
setwd("C:/Users/Mohammed/Desktop/EARL_Tutorial/Scrapping")
#as.matrix(testing, ncol = 25)
tableGoodHome <- as.data.frame(matrix(testing,ncol =25,byrow = T))
tableUsefulHome <- tableGoodHome[, c(1, 2, 3, 10, 12)]
tableUsefulHome <- cbind(tableUsefulHome, RefDateProper)
colnames(tableUsefulHome) <- c("Number", "Pos", "Name", "GT", "PP", "Date")
write.csv(tableUsefulHome , file = "Home.csv")

visiting <- as.data.frame(Full_data_txt)[576:1075,1]
#setwd("C:/Users/Mohammed/Desktop/EARL_Tutorial/Scrapping")
#as.matrix(visiting, ncol = 25)
tableVisiting <- as.data.frame(matrix(visiting,ncol =25,byrow = T))
tableUsefulVisiting <- tableVisiting[, c(1, 2, 3, 10, 12)]

tableUsefulVisiting <- cbind(tableUsefulVisiting, RefDateProper)
colnames(tableUsefulVisiting) <- c("Number", "Pos", "Name", "GT", "PP", "Date")

write.csv(tableUsefulVisiting , file = "Visiting.csv")

tableUsefulVisiting <- c()
tableUsefulHome <- c()
