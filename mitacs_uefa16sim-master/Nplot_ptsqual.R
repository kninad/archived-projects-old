
#setwd("~/Desktop/Link to bayes/workdir/reportcode/newprogs-thur/")
#load("./Nptsqual1.RData") # rwin,rsmw,rqrw,rrsw .. same for points

numsim <- 10000
groupnames <- c("A","B","C","D","E","F")
teams <- c("France", "Romania", "Albania", "Switzerland", "England", "Russia", "Wales", "Slovakia", "Germany", "Ukraine", "Poland", "Northern Ireland", "Spain", "Czech Republic", "Turkey", "Croatia", "Belgium", "Italy", "Repulic of Ireland", "Sweden", "Portugal", "Iceland", "Austria", "Hungary")
tnum <- 1:24
ranksteam <- c(8,18,31,10,3,9,28,19,1,14,17,33,2,15,22,12,5,6,23,16,4,27,11,20)

## Functions ####
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

## set vars ####
twin <- assignteam(rwin)
tsmw <- assignteam(rsmw) 
tqrw <- assignteam(rqrw) 
trsw <- assignteam(rrsw)

cwin <- countnum(twin)
csmw <- countnum(tsmw)
cqrw <- countnum(tqrw)
crsw <- countnum(trsw)




