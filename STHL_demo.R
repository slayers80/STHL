library(lme4)
library(car)
library(ggplot2)
library(MASS)

setwd("C:/Users/lwang/My Work/EEG_Experiments/Results/Ex012 (STHL Algorithm)")

data <- read.csv(file="STHL_dataframe.csv",head=TRUE,sep=",")

Anova(fm00 <- lm(Speech ~ Gender, data))

Anova(fm01 <- lm(Speech ~ EEG1+EEG2+EEG3 + ITD1+ITD2+ITD3 + HL1+HL2+HL3, data))

vif(fm01)

step <- stepAIC(fm03, direction="both")
step$anova # display results

Anova(fm02 <- lm(Speech ~ EEG3 + ITD1 + ITD3, data))


