#“What is your expected p&l after tossing a fair coin 1000 times, if when it lands heads, you receive a dollar and when it lands tails, you pay me a dollar?”
#We can easily derived the expected value of this game by using the fact that the coin toss is a binary outcome with equal probability. E(pnl) = 1 * p + -1 * (1-p) = 0

#to do a simulation for one path

 sequenceOfCoinTosses <- sample(c(-1,1), 1000, replace = TRUE)

# if we want to look at our cumulative p&l for this single run, we can use the cumsum() function and then plot() the result.

 plot(cumsum(sequenceOfCoinTosses), type = 'l')
## where cumsum is the cumulitive sum of all the coin tosses (assuming you win a dollar for either choice and lose one for the other or if a drunken man is walking on a straight line. 

 ##Now, let’s repeat this game 10000 times and see what our final p&l ends up being each time. I will store every final p&l value in a list() container and then use the unlist() command to create a vector that I can subsequently plot using the hist() function.

  # Create an empty list to store the results
 results <- list()
 for(i in 1:10000) {
     coinTosses   <- cumsum(sample(c(-1,1), 1000, replace = TRUE)) 
     results[[i]] <- coinTosses[length(coinTosses)]
 }
 
 # Unlist the list and create a histogram. Set a title and set the color and breaks
   hist(unlist(results), main = "Histogram of all the final p&l's",col = "lightblue", breaks = 100)
 
 # Place a vertical line at 0 with a width of 2 in order to show the average of the distribution
 abline(v = 0, col = "red", lwd = 2)

 ## source http://www.rfortraders.com/simulation-of-a-coin-toss-in-r/

