
#setwd("~/Desktop/Link to bayes/workdir/reportcode/method3-wed/")
#load("./Nqualres-1.RData") # "drawqual"  "firstdraw" "onlyqual"

numsim <- 10000
groupnames <- c("A","B","C","D","E","F")
teams <- c("France", "Romania", "Albania", "Switzerland", "England", "Russia", "Wales", "Slovakia", "Germany", "Ukraine", "Poland", "Northern Ireland", "Spain", "Czech Republic", "Turkey", "Croatia", "Belgium", "Italy", "Repulic of Ireland", "Sweden", "Portugal", "Iceland", "Austria", "Hungary")
teams2 <- c("France", "Romania", "Albania", "Switzerld.", "England", "Russia", "Wales", "Slovakia", "Germany", "Ukraine", "Poland", "N. Ireland", "Spain", "Czech Rp", "Turkey", "Croatia", "Belgium", "Italy", "Ireland", "Sweden", "Portugal", "Iceland", "Austria", "Hungary")
tnum <- 1:24


#ponlyq <- as.double(onlyqual/numsim)

plossq <- as.double(lossqual/numsim) * 100
pftsls <- as.double(firstlos/numsim) * 100

pdrawq <- as.double(drawqual/numsim) * 100
pftsdr <- as.double(firstdraw/numsim) * 100

pwinq <- as.double(winqual/numsim) * 100
pftswn <- as.double(firstwin/numsim) * 100

#change ylim value as needed.
#barplot(ponlyq, names.arg = tnum, ylim = c(0,1.0), xpd = FALSE, main = "Qualification from group stage", xlab = "Team number", ylab = "Probability")

barplot(pftsdr, names.arg = tnum, ylim = c(0,0.35), xpd = FALSE, main = " Drew first match", xlab = "Team number", ylab = "Probability")
barplot(pdrawq, names.arg = tnum, ylim = c(0,0.25), xpd = FALSE, main = "Drawing first match and qualifying", xlab = "Team number", ylab = "Probability")

barplot(pftswn, names.arg = tnum, ylim = c(0,0.65), xpd = FALSE, main = "Won first match", xlab = "Team number", ylab = "Probability")
barplot(pwinq, names.arg = tnum, ylim = c(0,0.65), xpd = FALSE, main = "Winning first match and qualifying", xlab = "Team number", ylab = "Probability")

barplot(pftsdr, names.arg = tnum, ylim = c(0,0.35), xpd = FALSE, main = "Lost first match", xlab = "Team number", ylab = "Probability")
barplot(pdrawq, names.arg = tnum, ylim = c(0,0.25), xpd = FALSE, main = "Losing first match and qualifying", xlab = "Team number", ylab = "Probability")

drawcond <- as.double(pdrawq / pftsdr)  # given that drawn
losscond <- as.double(plossq / pftsls) # given that lost
wincond <- as.double(pwinq / pftswn) #given that won

dr_qualcond <- as.double(pdrawq/ponlyq) #1st draw given qual
ls_qualcond <- as.double(plossq/ponlyq) #1st loss given qual
wn_qualcond <- as.double(pwinq/ponlyq)  #1st win given qual   





