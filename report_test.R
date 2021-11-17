library(ggplot2)

mydata <- mtcars

chartA <-  ggplot(mydata, aes(x = wt, y=qsec)) + 
  geom_point()

chartA2 <- ggplot(mydata, aes(x = mpg, y=hp)) +
  geom_point()
