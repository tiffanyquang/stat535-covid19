data = read.csv("master_state_data.csv", header= TRUE)
attach(data)

library(leaps)

resad <- regsubsets(positive_04102020 ~ total_test_results_04102020 + score + Density
                    + gdpRank + Airports + Pop + LandArea + Automobiles,
                    data = data, nbest = 3, method = "exhaustive")
par(cex.axis = 1.5, cex.lab = 1.5, mar = c(5,5,1,1))
plot(resad, scale = "adjr2")

summ <- summary(resad)
selected <- summ$which
selected
