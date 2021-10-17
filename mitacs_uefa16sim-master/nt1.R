#load("./Nptsqual1.RData") # rwin,rsmw,rqrw,rrsw .. same for points

findgrp <- function(tnum){
     x <- tnum%/%4
     y <- tnum%%4
     grpn <- 0
     if(y!=0){
         grpn <- (x+1)
     }else{
         grpn <- (x)
    }
    return(grpn)
}

numgrpvect <- function(wvect){
    l1 <- length(wvect)
    gvect <- vector(mode = "numeric", length = l1)
    for(i in 1:l1){
        gvect[i] <- findgrp(wvect[i]) 
    }
    return(gvect)
}

findteam <- function(rank) {
    tnum <- 0
    for(i in 1:24){
        if(rank == ranksteam[i] ){
            tnum <- i
            break
        }
    }
    return(tnum)
}

assignteam <- function(rvect){
    tvect <- vector(mode = "numeric", length = length(rvect))
    for (i in 1:length(tvect)){
        tvect[i] <- findteam(rvect[i])
    }
    return(tvect)
}

countnum <- function(tvect){
    countvect <- vector(mode = "numeric", length = 24)
    for(i in 1:24){
        countvect[i] <- sum(tvect == i)
    }
    return(countvect)
}

numsim <- 10000
groupnames <- c("A","B","C","D","E","F")
teams <- c("France", "Romania", "Albania", "Switzerland", "England", "Russia", "Wales", "Slovakia", "Germany", "Ukraine", "Poland", "Northern Ireland", "Spain", "Czech Republic", "Turkey", "Croatia", "Belgium", "Italy", "Repulic of Ireland", "Sweden", "Portugal", "Iceland", "Austria", "Hungary")
teams2 <- c("France", "Romania", "Albania", "Switzerld.", "England", "Russia", "Wales", "Slovakia", "Germany", "Ukraine", "Poland", "N. Ireland", "Spain", "Czech Rp", "Turkey", "Croatia", "Belgium", "Italy", "Ireland", "Sweden", "Portugal", "Iceland", "Austria", "Hungary")
tnum <- 1:24
ranksteam <- c(8,18,31,10,3,9,28,19,1,14,17,33,2,15,22,12,5,6,23,16,4,27,11,20)

twin <- assignteam(rwin)
tsmw <- assignteam(rsmw) 
tqrw <- assignteam(rqrw) 
trsw <- assignteam(rrsw)

winvect <- numgrpvect(twin) #groups of winners
smwvect <- numgrpvect(tsmw) #groups of finalists
qrwvect <- numgrpvect(tqrw)
rswvect <- numgrpvect(trsw)

wingrp <- vector(mode = "numeric", length = 6) #winner
smwgrp <- vector(mode = "numeric", length = 6) #finalist
qrwgrp <- vector(mode = "numeric", length = 6) #semi-finalist
rswgrp <- vector(mode = "numeric", length = 6) #quarter-finalist 
totvect <- vector(mode = "numeric", length = 6)

#obtain prob dist for grpnum ####
for(i in 1:6) { 
    wingrp[i] <- sum(winvect == i) / length(winvect)
    smwgrp[i] <- sum(smwvect == i) / length(smwvect)
    qrwgrp[i] <- sum(qrwvect == i) / length(qrwvect)
    rswgrp[i] <- sum(rswvect == i) / length(rswvect)
    totvect[i] <- sum(winvect == i)+sum(smwvect == i)+sum(qrwvect == i)+sum(rswvect == i)
}

#Count winners etc... ####
cwin <- countnum(twin)
csmw <- countnum(tsmw)
cqrw <- countnum(tqrw)
crsw <- countnum(trsw)

#qualifying to G.S####
pqual <- onlyqual2/100
sq <- sort(pqual , decreasing = TRUE, index.return = TRUE)
sqval <- sq$x
sqt <- teams2[sq$ix] 
tnum <- 1:24
#PLOT
#barplot(pqual, ylim = c(15,100), main = "Progress to Round of 16", ylab = "Percentage chance", xlab = "Team number", names.arg = tnum, xpd = FALSE)
#top-5
barplot(sqval, ylim = c(0,100), main = "Progress to Round of 16", ylab = "Percentage chance", xlab = "", names.arg = sqt, xpd = FALSE, las=2)

#prob is getting halved####
x1 <- crsw/onlyqual2 * 100
x2 <- cqrw/crsw * 100
x3 <- csmw/cqrw * 100
x4 <- cwin/csmw * 100
#overall performance prob.
xtot <- totvect/150000



