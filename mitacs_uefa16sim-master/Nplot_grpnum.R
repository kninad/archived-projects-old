
groupnames <- c("A","B","C","D","E","F")
#change ylim value as needed.
barplot(rswgrp, ylim = c(0.1,0.18), main = "Quarter-finalists from each group", ylab = "Density", xlab = "Groups",  col = "lightblue", names.arg = groupnames, xpd = FALSE)
barplot(qrwgrp, ylim = c(0.1,0.2), main = "Semi-finalists from each group", ylab = "Density", xlab = "Groups",  col = "lightblue", names.arg = groupnames, xpd = FALSE)
barplot(smwgrp, ylim = c(0.1,0.2), main = "Finalists from each group", ylab = "Density", xlab = "Groups",  col = "lightblue", names.arg = groupnames, xpd = FALSE)
barplot(wingrp, ylim = c(0.1,0.205), main = "Cup winners from each group", ylab = "Density", xlab = "Groups",  col = "lightblue", names.arg = groupnames, xpd = FALSE)

## For los/draw/win qual ####

