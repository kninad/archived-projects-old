
#setwd("~/Desktop/Link to bayes/workdir/reportcode/")
setwd("~/Downloads/newprogs-thur/")

######################################################
# RENAME RData file to be saved before running the program!!
######################################################

## Global variables - Fixed data ####

allmat <- read.csv("./matchups.csv") #all matches
#allmat <- read.csv("./qual1.csv") #all matches
#tabelo <- read.csv("./elotab.csv")

#prob. of different kinds on scores
rprob <- c(0.24852071, 0.24852071, 0.09467456, 0.19526627, 0.06508876, 0.01775148, 0.09467456, 0.02366864, 0.01183432)
dprob <- c(0.40000000, 0.48333333, 0.08333333, 0.03333333)

#teams <- c(ga, gb, gc, gd, ge, gf) # teams vector
teams <- c("France", "Romania", "Albania", "Switzerland", "England", "Russia", "Wales", "Slovakia", "Germany", "Ukraine", "Poland", "Northern Ireland", "Spain", "Czech Republic", "Turkey", "Croatia", "Belgium", "Italy", "Repulic of Ireland", "Sweden", "Portugal", "Iceland", "Austria", "Hungary" )
tmatx <- matrix(teams, nrow = 4, ncol = 6) # teams matrix
#telo <- c(1952, 1729, 1584, 1745, 1947, 1736, 1629, 1742, 2011, 1812, 1743, 1595, 1973, 1728, 1802, 1745, 1898, 1853, 1735, 1738, 1885, 1640, 1752, 1668)

#UEFA ranking of teams
ranksteam <- c(8,18,31,10,3,9,28,19,1,14,17,33,2,15,22,12,5,6,23,16,4,27,11,20)
trank <- matrix(ranksteam, nrow = 4, ncol = 6) # uefa ranks matrix

## Functions ####

findprob<-function(team1 , team2) {
    
    for(i in 1:276){
        pvec <- rep(0,3)
        
        if(allmat[i,1] == team1 && allmat[i,2] == team2){
            pvec[1] <- allmat[i,4] #win1
            pvec[2] <- allmat[i,6] #win2
            pvec[3] <- allmat[i,5] #draw
            break
            
        } else if(allmat[i,1] == team2 && allmat[i,2] == team1){
            pvec[1] <- allmat[i,6] #win1
            pvec[2] <- allmat[i,4] #win2
            pvec[3] <- allmat[i,5] #draw
            break
        }
    }
    return(as.double(pvec) )
}

# function to simulate result of game
simgame <- function(team1 , team2, n) {
    
    mval <- c(3,0,1) # values for RV for match
    vecprob <- findprob(team1, team2)
    simx <- sample(x = mval, size = n, replace = TRUE, prob = vecprob)
    return(simx) 
}

# function using above 2 utility functions to simulate all games of a group
#tgv <- rep(9,6)
groupsim <- function(grptm) {
    
    # To be in order: 1,3,1,2,1,2
    gamval <- rep(0,6) # for 6 matches in a group
    
    gamval[1] <- simgame(grptm[1], grptm[2], 1)
    gamval[2] <- simgame(grptm[3], grptm[4], 1)
    #tgv <<- gamval
    # For games 3,4
    modsimgame3<-function(team1, team2, n) { #not using pc
        mval <- c(3,0,1) # values for RV for match
        
        if(gamval[1] == gamval[2]){ #increase draw prob.
            
            temprob <- findprob(team1, team2)
            temprob[3] <- temprob[3] + (temprob[1]+temprob[2])/4
            temprob[1] <- 3/4 * temprob[1] 
            temprob[2] <- 3/4 * temprob[2] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
            
        }else if(gamval[1] > gamval[2]){
            
            temprob <- findprob(team1, team2)
            temprob[1] <- temprob[1] + temprob[2]/4
            temprob[2] <- 3/4 * temprob[2] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
            
        }else if(gamval[1] < gamval[2]){
            
            temprob <- findprob(team1, team2)
            temprob[2] <- temprob[2] + temprob[1]/4
            temprob[1] <- 3/4 * temprob[1] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
        }
    }
    
    modsimgame4<-function(team1, team2, n) {
        mval <- c(3,0,1) # values for RV for match
        
        if(gamval[1] == gamval[2]) {  #increase draw prob.
            
            temprob <- findprob(team1, team2)
            temprob[3] <- temprob[3] + (temprob[1]+temprob[2])/4
            temprob[1] <- 3/4 * temprob[1] 
            temprob[2] <- 3/4 * temprob[2] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
            
        }else if(gamval[1] < gamval[2]){
            
            temprob <- findprob(team1, team2)
            temprob[1] <- temprob[1] + temprob[2]/4
            temprob[2] <- 3/4 * temprob[2] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
            
        }else if(gamval[1] > gamval[2]){
            
            temprob <- findprob(team1, team2)
            temprob[2] <- temprob[2] + temprob[1]/4
            temprob[1] <- 3/4 * temprob[1] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
        }
    }
    
    gamval[3] <- modsimgame3(grptm[1], grptm[3], 1)
    gamval[4] <- modsimgame4(grptm[2], grptm[4], 1)
    #tgv <<- gamval
    
    # For games 5,6
    modsimgame5<-function(team1, team2, n) { 
        mval <- c(3,0,1) # values for RV for match
        g4 <- gamval[4]
        
        if(gamval[4] != 1){
            g4 <- 3 - gamval[4] #invert
        }
        
        
        if(gamval[3] == g4) { #increase draw prob.
            
            temprob <- findprob(team1, team2)
            temprob[3] <- temprob[3] + (temprob[1]+temprob[2])/4
            temprob[1] <- 3/4 * temprob[1] 
            temprob[2] <- 3/4 * temprob[2] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
            
        }else if(gamval[3] > g4) {
            
            temprob <- findprob(team1, team2)
            temprob[1] <- temprob[1] + temprob[2]/4
            temprob[2] <- 3/4 * temprob[2] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
            
        }else if(gamval[3] < g4) {
            
            temprob <- findprob(team1, team2)
            temprob[2] <- temprob[2] + temprob[1]/4
            temprob[1] <- 3/4 * temprob[1] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
        }
    }
    
    modsimgame6<-function(team1, team2, n) { 
        mval <- c(3,0,1) # values for RV for match
        g4 <- gamval[4]
        
        if(gamval[4] != 1){
            g4 <- 3 - gamval[4] #invert
        }       
        
        if(gamval[3] == g4){ #increase draw prob.
            
            temprob <- findprob(team1, team2)
            temprob[3] <- temprob[3] + (temprob[1]+temprob[2])/4
            temprob[1] <- 3/4 * temprob[1] 
            temprob[2] <- 3/4 * temprob[2] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
            
        }else if(gamval[3] < g4){
            
            temprob <- findprob(team1, team2)
            temprob[1] <- temprob[1] + temprob[2]/4
            temprob[2] <- 3/4 * temprob[2] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
            
        }else if(gamval[3] > g4){
            
            temprob <- findprob(team1, team2)
            temprob[2] <- temprob[2] + temprob[1]/4
            temprob[1] <- 3/4 * temprob[1] 
            
            simx <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
            
            return(simx)
        }
    }
    
    gamval[5] <- modsimgame5(grptm[1], grptm[4], 1)
    gamval[6] <- modsimgame6(grptm[2], grptm[3], 1)
    #tgv <<- gamval
    return(gamval)    
}

# function to simulate results entire group stage of cup
allgrpsim <- function(tmat) {
    grpval <- matrix(rep(0,36), nrow = 6, ncol = 6)
    for(i in 1:6) {
        temp <- tmat[,i]
        grpval[,i] <- groupsim(temp)
    }
    return(grpval)
}

# function to calculate points of ONE group
calcpoints <- function(resmat) {
    
    pts <- rep(0,4)
    
    ############################# MATCH-1
    if(resmat[1] == 3) {
        
        pts[1] <- pts[1] + 3
    }
    
    if(resmat[1] == 0) {
        
        pts[2] <- pts[2] + 3
    }
    
    if(resmat[1] == 1) {
        pts[1] <- pts[1] + 1
        pts[2] <- pts[2] + 1
    }    
    
    ############################# MATCH-2
    if(resmat[2] == 3) {
        
        pts[3] <- pts[3] + 3
    }
    
    if(resmat[2] == 0) {
        
        pts[4] <- pts[4] + 3
    }
    
    if(resmat[2] == 1) {
        pts[3] <- pts[3] + 1
        pts[4] <- pts[4] + 1
    }    
    ############################# MATCH-3
    if(resmat[3] == 3) {
        
        pts[1] <- pts[1] + 3
    }
    
    if(resmat[3] == 0) {
        
        pts[3] <- pts[3] + 3
    }
    
    if(resmat[3] == 1) {
        pts[1] <- pts[1] + 1
        pts[3] <- pts[3] + 1
    }    
    
    ############################# MATCH-4
    if(resmat[4] == 3) {
        
        pts[2] <- pts[2] + 3
    }
    
    if(resmat[4] == 0) {
        
        pts[4] <- pts[4] + 3
    }
    
    if(resmat[4] == 1) {
        pts[4] <- pts[4] + 1
        pts[2] <- pts[2] + 1
    }    
    
    ############################# MATCH-5
    if(resmat[5] == 3) {
        
        pts[1] <- pts[1] + 3
    }
    
    if(resmat[5] == 0) {
        
        pts[4] <- pts[4] + 3
    }
    
    if(resmat[5] == 1) {
        pts[1] <- pts[1] + 1
        pts[4] <- pts[4] + 1
    }    
    
    ############################# MATCH-6
    if(resmat[6] == 3) {
        
        pts[2] <- pts[2] + 3
    }
    
    if(resmat[6] == 0) {
        
        pts[3] <- pts[3] + 3
    }
    
    if(resmat[6] == 1) {
        pts[2] <- pts[2] + 1
        pts[3] <- pts[3] + 1
    }    
    
    return(pts)
}

# function to calculate points of ALL group
allpoints <- function(simln) { 
    grpoints <- matrix(rep(0,24), nrow = 4, ncol = 6)
    for(i in 1:6) {
        grpoints[,i] <- calcpoints(simln[,i])
    }
    return(grpoints)
}

#simulate score of a match from result value (3,1,0)
simscore <- function(result){
    
    rtypes <- c(1,2,3,4,5,6,7,8,9)
    dtypes <- c(1,2,3,4)
    
    gst1 <- 0
    gst2 <- 0
    gat1 <- 0
    gat2 <- 0
    
    if(result == 1){ #draw
        simD <- sample(x = dtypes, size = 1, replace = TRUE, prob = dprob)   
        
        if(simD == 1){ #0-0
            gst1 <- gst1 + 0
            gst2 <- gst2 + 0
            gat1 <- gat1 + 0
            gat2 <- gat2 + 0
        }
        if(simD == 2){ # 1-1
            gst1 <- gst1 + 1
            gst2 <- gst2 + 1
            gat1 <- gat1 + 1
            gat2 <- gat2 + 1
        }
        if(simD == 3){ # 2-2
            gst1 <- gst1 + 2
            gst2 <- gst2 + 2
            gat1 <- gat1 + 2
            gat2 <- gat2 + 2
        }
        if(simD == 4){ # 3-3
            gst1 <- gst1 + 3
            gst2 <- gst2 + 3
            gat1 <- gat1 + 3
            gat2 <- gat2 + 3
        }
    } else{ #no-draw
        simR <- sample(x = rtypes, size = 1, replace = TRUE, prob = rprob)      
        
        st1 <- 0
        st2 <- 0
        at1 <- 0
        at2 <- 0
        
        if(simR == 1){ #1-0
            st1 <- st1 + 1
            st2 <- st2 + 0
            at1 <- at1 + 0
            at2 <- at2 + 1
        } else        if(simR == 2){ #2-1
            st1 <- st1 + 2
            st2 <- st2 + 1
            at1 <- at1 + 1
            at2 <- at2 + 2
        }else        if(simR == 3){ #3-2
            st1 <- st1 + 3
            st2 <- st2 + 2
            at1 <- at1 + 2
            at2 <- at2 + 3
        }else        if(simR == 4){ #2-0
            st1 <- st1 + 2
            st2 <- st2 + 0
            at1 <- at1 + 0
            at2 <- at2 + 2
        }else        if(simR == 5){ #3-1
            st1 <- st1 + 3
            st2 <- st2 + 1
            at1 <- at1 + 1
            at2 <- at2 + 3
        }else        if(simR == 6){ #4-2
            st1 <- st1 + 4
            st2 <- st2 + 2
            at1 <- at1 + 2
            at2 <- at2 + 4
        }else        if(simR == 7){ #3-0
            st1 <- st1 + 3
            st2 <- st2 + 0
            at1 <- at1 + 0
            at2 <- at2 + 3
        }else        if(simR == 8){ #4-1
            st1 <- st1 + 4
            st2 <- st2 + 1
            at1 <- at1 + 1
            at2 <- at2 + 4
        }else        if(simR == 9){ #4-0
            st1 <- st1 + 4
            st2 <- st2 + 0
            at1 <- at1 + 0
            at2 <- at2 + 4
        }
        
        if(result == 3){
            gst1 <- st1
            gat1 <- at1
            gst2 <- st2
            gst2 <- at2
            
        } else if(result == 0){ #swap values
            gst1 <- st2
            gat1 <- at2
            gst2 <- st1
            gst2 <- at1
        }  
        
    } # no-draw ends
    scorvectemp <- c(gst1, gst2, gat1, gat2) 
    
    return(scorvectemp)
} 

# function to calculate scores of group
calcgoals <- function(resmat) {
    
    gscr <- rep(0,4)
    gsag <- rep(0,4)
    
    for (i in 1:6){
        
        tempvect <- simscore(resmat[i])
        
        if(i == 1){ #match-1
            gscr[1] <- tempvect[1]
            gscr[2] <- tempvect[2]
            gsag[1] <- tempvect[3]
            gsag[2] <- tempvect[4]
            
        }else if(i ==2){ #match-2
            gscr[3] <- tempvect[1]
            gscr[4] <- tempvect[2]
            gsag[3] <- tempvect[3]
            gsag[4] <- tempvect[4]
            
        }else if(i==3){
            gscr[1] <- tempvect[1]
            gscr[3] <- tempvect[2]
            gsag[1] <- tempvect[3]
            gsag[3] <- tempvect[4]
            
        }else if(i==4){
            gscr[2] <- tempvect[1]
            gscr[4] <- tempvect[2]
            gsag[2] <- tempvect[3]
            gsag[4] <- tempvect[4]
            
        }else if(i==5){
            gscr[1] <- tempvect[1]
            gscr[4] <- tempvect[2]
            gsag[1] <- tempvect[3]
            gsag[4] <- tempvect[4]
            
        }else if(i == 6){
            gscr[2] <- tempvect[1]
            gscr[3] <- tempvect[2]
            gsag[2] <- tempvect[3]
            gsag[3] <- tempvect[4]
        }
    } #loop ends
    
    goal_matrix <- matrix(rep(0,8), nrow = 4, ncol = 2)
    goal_matrix[,1] <- gscr
    goal_matrix[,2] <- gscr - gsag
    return(goal_matrix) #col-1: g.s ; col-2: g.a
    
}

# all goals simulation stored in a big matrix
allgoals <- function(simln){ #pass complete simulation as argument
    clmatrix <- matrix(rep(0,48), nrow = 4, ncol = 12)
    for(i in 1:6) {
        tempmatrix <- calcgoals(simln[,i])
        clmatrix[,i] <- tempmatrix[,1]
        clmatrix[,i+6] <- tempmatrix[,2]
    }
    return(clmatrix)
}

#breaking up cmatrix for goals scored and difference
allglscr <- function(clmatrix){
    goalscr <- matrix(rep(0,24), nrow = 4, ncol = 6)
    for(i in 1:6){ 
        goalscr[,i] <- clmatrix[,i]
    }
    return(goalscr)
} 

allgldif <- function(clmatrix){
    goaldif <- matrix(rep(0,24), nrow = 4, ncol = 6)
    for(i in 1:6){ 
        goaldif[,i] <- clmatrix[,i+6]
    }
    return(goaldif)
} 


#big function to order a group
ordgroup <- function(g,glscr,gldif,matpts,simat) {
    #g <- 4 # group no.
    #glscr
    #gldif
    
    grpts <- matpts[,g]
    res <- simat[,g]
    
    id <- sort(grpts, decreasing = TRUE, index.return = TRUE)
    pts <- id$x
    # NOTE THAT pts already in sorted order
    
    if(length(unique(pts)) == length(pts)) { #all diff
        return(id)
    }
    
    
    #2same # ti is at higher posn in table (lower index) than tj
    h2h <- function(ti,tj,res){ 
        
        retval <- 0 # 0=no change, 1 = draw, go further, 2 = invert
        
        if(ti == 1 && tj == 2){     #res1
            if(res[1] == 0){
                retval <- 2
            }else if(res[1] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        if(ti == 2 && tj == 1){
            if(res[1] == 3){
                retval <- 2
            }else if(res[1] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        if(ti == 3 && tj == 4){     #res2
            if(res[2] == 0){
                retval <- 2
            }else if(res[2] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        if(ti == 4 && tj == 3){
            if(res[2] == 3){
                retval <- 2
            }else if(res[2] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        if(ti == 1 && tj == 3){     #res3
            if(res[3] == 0){
                retval <- 2
            }else if(res[3] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        if(ti == 3 && tj == 1){
            if(res[3] == 3){
                retval <- 2
            }else if(res[3] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        if(ti == 2 && tj == 4){     #res4
            if(res[4] == 0){
                retval <- 2
            }else if(res[4] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        if(ti == 4 && tj == 2){
            if(res[4] == 3){
                retval <- 2
            }else if(res[4] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        if(ti == 1 && tj == 4){     #res5
            if(res[5] == 0){
                retval <- 2
            }else if(res[5] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        if(ti == 4 && tj == 1){
            if(res[5] == 3){
                retval <- 2
            }else if(res[5] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        
        if(ti == 2 && tj == 3){     #res6
            if(res[6] == 0){
                retval <- 2
            }else if(res[6] == 1){
                retval <- 1
            }
            return(retval)
        }
        
        if(ti == 3 && tj == 2){
            if(res[6] == 3){
                retval <- 2
            }else if(res[6] == 1){
                retval <- 1
            }
            return(retval)
        }
    }
    
    posn2 <- function(i,j,res,id) {
        
        ti <- id$ix[i]
        tj <- id$ix[j]
        
        val <- h2h(ti,tj,res)	
        
        if(val == 2){ # INVERT
            valtemp<- tj #swap 
            tj <- ti
            ti <- valtemp
            
        }
        
        if(val == 1){ #DRAW
            
            if(gldif[tj,g] > gldif[ti,g]){
                valtemp<- tj #swap 
                tj <- ti
                ti <- valtemp
                
            }else if(gldif[tj,g] == gldif[ti,g]){
                if(glscr[tj,g] > glscr[ti,g]){
                    valtemp<- tj #swap 
                    tj <- ti
                    ti <- valtemp
                    
                }else if( glscr[tj,g] == glscr[ti,g] ) {
                    if(trank[ti,g] > trank[tj,g]){ # only then swap
                        valtemp<- tj #swap 
                        tj <- ti
                        ti <- valtemp
                    }
                }
            }
        }
        
        #after swapping and tinkering, insert back original values
        id$ix[i] <- ti
        id$ix[j] <- tj
        
        return(id) #store this id in a new list
    } #resolving between 2 equal points
    
    if( (pts[1] == pts[2]) && (pts[2] != pts[3]) && (pts[3] != pts[4]) ) {
        nid <- posn2(1,2,res,id)
        return(nid)
    }
    
    if( (pts[3] == pts[2]) && (pts[2] != pts[1]) && (pts[1] != pts[4]) ){
        nid <- posn2(2,3,res,id)
        return(nid)
    }
    
    if( (pts[3] == pts[4]) && (pts[2] != pts[3]) && (pts[1] != pts[2]) ){
        nid <- posn2(3,4,res,id)
        return(nid)
    }
    
    if( pts[1] == pts[2] && pts[3] == pts[4] ){
        nid1 <- posn2(1,2,res,id)
        nid2 <- posn2(3,4,res,nid1)
        return(nid2)
    }
    
    #3same
    posn3 <- function(i,j,k,id,glscr,gldif){ # i>j>k in terms of posn
        
        ti <- id$ix[i]
        tj <- id$ix[j]
        tk <- id$ix[k]
        
        tgd <- c(gldif[i,g], gldif[j,g], gldif[k,g])
        
        gdid <- sort(tgd, decreasing = TRUE, index.return = TRUE)
        gd <- gdid$x
        
        if(length(unique(gd)) == length(gd)) { # ALL DISTINCT
            
            ti <- gdid$ix[1]
            tj <- gdid$ix[2]
            tk <- gdid$ix[3]
            
        } else if(length(unique(gd)) == 1){ # ALL SAME, go to goal scored
            
            tgs <- c(glscr[i,g], glscr[j,g], glscr[k,g])
            gsid <- sort(tgs, decreasing = TRUE, index.return = TRUE)
            gs <- gsid$x
            
            if(length(unique(gs)) == length(gs)) { # ALL DISTINCT
                
                ti <- gsid$ix[1]
                tj <- gsid$ix[2]
                tk <- gsid$ix[3]
                
            } else { # ALL SAME 
                
                tempranks <- c(trank[i,g], trank[j,g], trank[k,g])
                trid <- sort(tempranks, decreasing = FALSE, index.return = TRUE)
                ti <- trid$ix[1]
                tj <- trid$ix[2]
                tk <- trid$ix[3]
            }
            
        } else { #2-same
            
            if(gd[1] != gd[2] && gd[2] == gd[3]){
                
                if(trank[2,g] > trank[3,g]){
                    tj <- gdid$ix[3]
                    tk <- gdid$ix[2]
                } else{
                    tj <- gdid$ix[2]
                    tk <- gdid$ix[3]
                }
                
                ti <- gdid$ix[1]
            }
            
            if(gd[3] != gd[1] && gd[1] == gd[2]){
                
                if(trank[1,g] > trank[2,g]){
                    ti <- gdid$ix[2]
                    tj <- gdid$ix[1]
                } else{
                    ti <- gdid$ix[1]
                    tj <- gdid$ix[2]
                }
                
                tk <- gdid$ix[3]
            }
        }
        
        #after swapping and tinkering, insert back original values
        id$ix[i] <- ti
        id$ix[j] <- tj
        id$ix[k] <- tk
        
        return(id) #store this id in a new list
    } 
    
    if(pts[1]==pts[2] && pts[2]==pts[3] && pts[3]!=pts[4]){
        nid <- posn3(1,2,3,id,glscr,gldif)
        return(nid)
    }
    
    if(pts[1]!=pts[2] && pts[2]==pts[3] && pts[3]==pts[4]){
        nid <- posn3(2,3,4,id,glscr,gldif)
        return(nid)
    }
    
    
    #all same
    posn4 <- function(id,glscr,gldif){
        
        t1 <- id$ix[1]
        t2 <- id$ix[2]
        t3 <- id$ix[3]
        t4 <- id$ix[4]
        
        tgd <- c(gldif[1,g], gldif[2,g], gldif[3,g], gldif[4,g])
        
        gdid <- sort(tgd, decreasing = TRUE, index.return = TRUE)
        gd <- gdid$x
        
        if(length(unique(gd)) == length(gd)) { # ALL DISTINCT
            
            t1 <- gdid$ix[1]
            t2 <- gdid$ix[2]
            t3 <- gdid$ix[3]
            t4 <- gdid$ix[4]
            
        } else { # ALL SAME, go to goal scored
            
            tgs <- c(glscr[1,g], glscr[2,g], glscr[3,g], glscr[4,g])
            gsid <- sort(tgs, decreasing = TRUE, index.return = TRUE)
            gs <- gsid$x
            
            if(length(unique(gs)) == length(gs)) { # ALL DISTINCT
                
                t1 <- gsid$ix[1]
                t2 <- gsid$ix[2]
                t3 <- gsid$ix[3]
                t4 <- gsid$ix[4]
                
            } else {  
                
                tempranks <- c(trank[1,g], trank[2,g], trank[3,g], trank[4,g])
                trid <- sort(tempranks, decreasing = FALSE, index.return = TRUE)
                t1 <- trid$ix[1]
                t2 <- trid$ix[2]
                t3 <- trid$ix[3]
                t4 <- trid$ix[4]
            }
            
        } 
        
        id$ix[1] <- t1
        id$ix[2] <- t2
        id$ix[3] <- t3
        id$ix[4] <- t4
        
        return(id)
    }
    
    if(length(unique(pts)) == 1) {
        nid <- posn4(1,2,3,4,res,id)
        return(nid)
    }
    
} 


#order and return final pos for all groups (matpts = finpts)
allfinpos <- function(glscr,gldif,matpts,simat){
    
    finpos <- matrix(rep(0,36), nrow = 4, ncol = 6)
    
    for (i in 1:6) {
        finpos[,i] <- ordgroup(i,glscr,gldif,matpts,simat)$ix
    }
    
    return(finpos)
}

#var1 <- 7
#third place teams qual
#After final ordering pick out 3rd placed teams and select for qualification
thirdplace <- function(finalpos,finpts,glscr,gldif){
    
    t3place <- finalpos[3,] #team-ids
    t3pts <- rep(0,6) #points of 3-place teams
    for(i in 1:6) {
        tmpid <- t3place[i]
        t3pts[i] <- finpts[tmpid,i] 
    }
    
    pts3sort <- sort(t3pts, decreasing = TRUE, index.return = TRUE)
    pts3val <- pts3sort$x
    pts3ids <- pts3sort$ix #ids(group no.) after sorting 
    
    if(length(unique(pts3val)) == length(pts3val)) { #all diff
        return(pts3sort)
    }
    
    if(length(unique(pts3val)) == 1){ #all same (check G.D)
        
        t3gd <- rep(0,6)
        for(i in 1:6){
            tmpid1 <- t3place[i]
            var1 <<- tmpid#DEBUG
            t3gd[i] <- gldif[tmpid1, i]
        }
        t3gdsort <- sort(t3gd, decreasing = TRUE, index.return = TRUE)
        
        if(length(unique(t3gdsort$x)) == length(t3gdsort$x) ){
            return(t3gdsort)
            
        } else { #all same (check G.S)
            
            t3gs <- rep(0,6)
            for(i in 1:6){
                tmpid <- t3place[i]
                t3gs[i] <- glscr[tmpid, i]
            }
            t3gssort <- sort(t3gs, decreasing = TRUE, index.return = TRUE)
            
            if(length(unique(t3gssort$x)) == length(t3gssort$x) ){
                return(t3gssort)
                
            } else { # (check rank)
                
                t3rank <- rep(0,6)
                for(i in 1:6){
                    tmpid <- t3place[i]
                    t3rank[i] <- trank[tmpid, i]
                }
                
                t3rksort <- sort(t3rank, decreasing = TRUE, index.return = TRUE)
                return(t3rksort)
            }
        }
        
    } else {
        
        if(pts3val[4] != pts3val[5] && length(unique(pts3val[1:4])) == length(pts3val[1:4]) ){
            return(pts3sort)
        }
        
        if( length(unique(pts3val[1:4])) == 1 ) {
            t4rp <- pts3ids[1:4]
            t4gd <- rep(0,6)
            for(i in 1:6){
                if(i == t4rp[1] || i == t4rp[2] || i == t4rp[3] || i == t4rp[4]) {
                    tmpid2 <- t3place[i]
                    t4gd[i] <- gldif[tmpid2, i]
                } else{
                    t4gd[i] <- -15
                }
            }
            t4gdsort <- sort(t4gd, decreasing = TRUE, index.return = TRUE)
            
            if(length(unique(t4gdsort$x)) == length(t4gdsort$x) ){
                
                #(old)pts3sort$ix[1:4] <- t4gdsort$ix
                return(t4gdsort)
                
            } else { #all same (check G.S)
                t4rp <- pts3ids[1:4]
                t4gs <- rep(0,6)
                for(i in 1:6){
                    if(i == t4rp[1] || i == t4rp[2] || i == t4rp[3] || i == t4rp[4]) {
                        tmpid <- t3place[i]
                        t4gs[i] <- glscr[tmpid, i]
                    } else{
                        t4gs[i] <- -1
                    }
                }
                t4gssort <- sort(t4gs, decreasing = TRUE, index.return = TRUE)
                
                if(length(unique(t4gssort$x)) == length(t4gssort$x) ){
                    #(old)pts3sort$ix[1:4] <- t4gssort$ix
                    return(t4gssort)
                    
                } else { # (check rank)
                    t4rp <- pts3ids[1:4]
                    t4rank <- rep(0,6)
                    for(i in 1:6){
                        if(i == t4rp[1] || i == t4rp[2] || i == t4rp[3] || i == t4rp[4]) {
                            tmpid <- t3place[i]
                            t4rank[i] <- trank[tmpid, i]
                        } else{
                            t4gs[i] <- -2
                        }
                    }
                    
                    t4rksort <- sort(t4rank, decreasing = TRUE, index.return = TRUE)
                    #certainly all unique 
                    #(old)pts3sort$ix[1:4] <- t4rksort$ix
                    return(t4rksort)
                }
            }
            
        } else {
            
            return(pts3sort) #can insert more checks here
        }
    }
    
}

#qual status for all teams
qualmat <- function(finalpos, fint3ids){
    gsqual <- matrix(rep(0,36), nrow = 4, ncol = 6)
    #0=not qualified ; 1=qualified to the knockouts
    
    for(i in 1:6) {
        t1 <- finalpos[1,i]
        t2 <- finalpos[2,i]
        t3 <- finalpos[3,i]
        gsqual[t1,i] <- 1
        gsqual[t2,i] <- 1
        
        if( i==fint3ids[1] || i==fint3ids[2] || i==fint3ids[3] || i==fint3ids[4] ){
            gsqual[t3,i] <- 1
        }
    }
    return(gsqual)
}

#final table with team names
fintabname <- function(finalpos){
    finaltem <- matrix(rep(0,36), nrow = 4, ncol = 6)
    
    for(j in 1:6){
        for(i in 1:4){
            finaltem[i,j] <- tmatx[ finalpos[i,j] , j ]
        }
    }
    return(finaltem)
}


findpts <- function(tm) {
    
    pts <- -1 # initial val.
    
    temp <- 0
    
    for(i in 1:24){
        if(teams[i] == tm){
            temp <- i
            break
        }
    }
    
    x <- temp%/%4
    y <- temp%%4
    grpn <- 0
    tnum <- y
    
    if(y!=0){
        grpn <- (x+1)
    }else{
        grpn <- (x)
        tnum <- 4
    }
    
    return(finpts[tnum,grpn])
}

findposn <- function(tm) {
    
    pts <- 0 # initial val.
    temp <- 0
    for(i in 1:24){
        if(teams[i] == tm){
            temp <- i
            break
        }
    }
    x <- temp%/%4
    y <- temp%%4
    grpn <- 0
    tnum <- y
    if(y!=0){
        grpn <- (x+1)
    }else{
        grpn <- (x)
        tnum <- 4
    }
    
    return(finalpos[tnum,grpn])
}


#simulate a K.O game
simkogame <- function(t1,t2){ #returns results of a match
    mval <- c(3,0,1) # values for RV for match
    tres <- 4 # garbage value
    n <- 1
    p1 <- 0.605 # penalty prob. win for team going 1st
    p2 <- 0.395 # penalty prob. win for team going 2nd
    
    pts1 <- findposn(t1) #position in group
    pts2 <- findposn(t2) #position in group. Don't mind the variable name
    
    if(pts1 == pts2) { 
        temprob <- findprob(t1, t2)
        tres <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
        
    }else if(pts1 > pts2) {
        temprob <- findprob(t1, t2)
        temprob[1] <- temprob[1] + (temprob[2] * (0.3))
        temprob[2] <- (0.7) * temprob[2] 
        tres <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
        
    }else if(pts1 < pts2) {
        temprob <- findprob(t1, t2)
        temprob[2] <- temprob[2] + (temprob[1] * (0.3))
        temprob[1] <- (0.7) * temprob[1] 
        tres <- sample(x = mval, size = n, replace = TRUE, prob = temprob)
    } 
    
    if(tres == 1){ #DRAW ---> PENALTIES
        pfirst <- sample(1:2, size=1, prob = c(0.5,0.5)) #who will go first
        if(pfirst == 2){ #t2 goes first
            tres <- sample(c(0,3), size = 1, prob = c(p1,p2)) #t2 has higher prob
            
        } else if(pfirst == 1){ #t1 goes first
            tres <- sample(c(3,0), size = 1, prob = c(p1,p2)) #t1 has higher prob
        }
    }
    
    if(tres == 3){
        return(t1)
        
    } else if(tres == 0){
        return(t2)
    }
} #return team name


#gives r16winners
allr16win <- function(finaltem,fint3ids){
    
    srt3id <- sort(fint3ids) # for ease of checking
    
    nt1 <- c( finaltem[2,1], finaltem[1,6], finaltem[1,5], finaltem[2,2] )
    nt2 <- c( finaltem[2,3], finaltem[2,5], finaltem[2,4], finaltem[2,6] )
    
    vt1 <- c( finaltem[1,4], finaltem[1,2], finaltem[1,3], finaltem[1,1] )
    #15 cases for vt2
    {#15 cases for vt2
        if( all(srt3id == c(1,2,3,4)) ){
            vt2 <- c(finaltem[3,2],finaltem[3,4],finaltem[3,1],finaltem[3,3])
            
        } else if( all(srt3id == c(1,2,3,5))){
            vt2 <- c(finaltem[3,5],finaltem[3,1],finaltem[3,2],finaltem[3,3])
            
        } else if( all(srt3id == c(1,2,3,6)) ){
            vt2 <- c(finaltem[3,6],finaltem[3,1],finaltem[3,2],finaltem[3,3])
            
        } else if( all(srt3id == c(1,2,4,5)) ){
            vt2 <- c(finaltem[3,5],finaltem[3,1],finaltem[3,2],finaltem[3,4])
            
        } else if( all(srt3id == c(1,2,4,6)) ){
            vt2 <- c(finaltem[3,6],finaltem[3,1],finaltem[3,2],finaltem[3,4])
            
        } else if( all(srt3id == c(1,2,5,6)) ){
            vt2 <- c(finaltem[3,6],finaltem[3,1],finaltem[3,2],finaltem[3,5])
            
        } else if( all(srt3id == c(1,3,4,5)) ){
            vt2 <- c(finaltem[3,5],finaltem[3,4],finaltem[3,1],finaltem[3,3])
            
        } else if( all(srt3id == c(1,3,4,6)) ){
            vt2 <- c(finaltem[3,6],finaltem[3,4],finaltem[3,1],finaltem[3,3])
            
        } else if( all(srt3id == c(1,3,5,6)) ){
            vt2 <- c(finaltem[3,5],finaltem[3,1],finaltem[3,6],finaltem[3,3])
            
        } else if( all(srt3id == c(1,4,5,6)) ){
            vt2 <- c(finaltem[3,5],finaltem[3,1],finaltem[3,6],finaltem[3,4])
            
        } else if( all(srt3id == c(2,3,4,5)) ){
            vt2 <- c(finaltem[3,5],finaltem[3,4],finaltem[3,2],finaltem[3,3])
            
        } else if( all(srt3id == c(2,3,4,6)) ){
            vt2 <- c(finaltem[3,6],finaltem[3,4],finaltem[3,2],finaltem[3,3])
            
        } else if( all(srt3id == c(2,3,5,6)) ){
            vt2 <- c(finaltem[3,6],finaltem[3,3],finaltem[3,2],finaltem[3,5])
            
        } else if( all(srt3id == c(2,4,5,6)) ){
            vt2 <- c(finaltem[3,6],finaltem[3,4],finaltem[3,2],finaltem[3,5])
            
        } else if( all(srt3id == c(3,4,5,6)) ){
            vt2 <- c(finaltem[3,5],finaltem[3,4],finaltem[3,6],finaltem[3,3])
        }
    } 
    # vt2 defned
    
    r16win <- vector(mode = "character", length = 8) #RounOf16
    for(i in 1:8){
        if(i %% 2 == 1) {
            tmp <- i%/% 2 + 1
            r16win[i] <- simkogame(nt1[tmp] , nt2[tmp])    
            
        }else if(i %% 2 == 0) {
            tmp <- i%/% 2 
            r16win[i] <- simkogame(vt1[tmp] , vt2[tmp])
        }
    }
    return(r16win)
}

#gives quater-final winners
allqrtwin <- function(r16win){
    
    qrtwin <- vector(mode = "character", length = 4) #Quarterfinals
    for(i in 1:4){
        qrtwin[i] <- simkogame( r16win[2*i-1] , r16win[2*i] )
    }
    return(qrtwin)   
}

#gives semi-finals winners
allsemwin <- function(qrtwin) {
    semwin <- vector(mode = "character", length = 2) #Semi-finals
    for(i in 1:2){
        semwin[i] <- simkogame( qrtwin[2*i-1] , qrtwin[2*i] ) #Finals 
    }
    return(semwin)
}

### Majaa ####
numsim <- 10000

onlyqual <- rep(0,24)

firstdraw <- rep(0,24)
drawqual <- rep(0,24)

firstwin <- rep(0,24)
winqual <- rep(0,24)

firstlos <- rep(0,24)
lossqual <- rep(0,24)

for(k in 1:numsim) {
    
    #do entire simulation so that values are assigned.
    sim1 <- allgrpsim(tmatx) # first simulation of Group Stage
    finpts <- allpoints(sim1) #final points
    cmatrix <- allgoals(sim1) #big matrix
    goalscr <- allglscr(cmatrix)
    goaldif <- allgldif(cmatrix) 
    ##### group ordering and qualification to KO 
    finalpos <- allfinpos(goalscr, goaldif, finpts, sim1)
    fint3 <- thirdplace(finalpos, finpts, goalscr, goaldif)
    fint3ids <- fint3$ix[1:4] #the group ids from which the 3-place teams are selected.
    #qual. matrix
    gsqual <- qualmat(finalpos, fint3ids)
    #final pos with team names 
    #finaltem <- fintabname(finalpos) 
    # K.O rounds 
    #winners of r16 
    #r16win <- allr16win(finaltem , fint3ids) 
    #gives quater-final winners
    #qrtwin <- allqrtwin(r16win)
    #gives semi-finals winners
    #semwin <- allsemwin(qrtwin)
    #gives the overall winner
    #finalwin <- simkogame(semwin[1], semwin[2])
    
    #call inference functions--change some global variables
    for(i in 1:24) {
        grpn <- 0
        tnum <- 0
        x <- i%/%4
        y <- i%%4
        if(y!=0){
            grpn <- (x+1)
            tnum <- y
        }else{
            grpn <- (x)
            tnum <- 4
        }
        
        resqual <- gsqual[tnum,grpn]
        res1 <- sim1[1,grpn]
        res2 <- sim1[2,grpn]
        
        if(resqual == 1){
            onlyqual[i] <- onlyqual[i] + 1
        }
        
        #DRAW
        if(res1 == 1 && (tnum == 1 || tnum == 2) ) {
            firstdraw[i] <- firstdraw[i] + 1
            if(resqual == 1){
                drawqual[i] <- drawqual[i] + 1
            }
        }
        
        if(res2 == 1 && (tnum == 3 || tnum ==4) ){
            firstdraw[i] <- firstdraw[i] + 1
            if(resqual == 1){
                drawqual[i] <- drawqual[i] + 1
            }
        }
        
        #WIN
        if(res1 == 3 && tnum == 1){
            firstwin[i] <- firstwin[i] + 1
            if(resqual == 1){
                winqual[i] <- winqual[i] + 1
            }
        }else if(res1 == 0 && tnum == 2){
            firstwin[i] <- firstwin[i] + 1
            if(resqual == 1){
                winqual[i] <- winqual[i] + 1
            }
        }else if(res2 == 3 && tnum == 4){
            firstwin[i] <- firstwin[i] + 1
            if(resqual == 1){
                winqual[i] <- winqual[i] + 1
            }
        }else if(res2 == 0 && tnum == 3){
            firstwin[i] <- firstwin[i] + 1
            if(resqual == 1){
                winqual[i] <- winqual[i] + 1
            }
        }
        
        #LOSS
        if(res1 == 3 && tnum == 2){
            firstlos[i] <- firstlos[i] + 1
            if(resqual == 1){
                lossqual[i] <- lossqual[i] + 1
            }
        }else if(res1 == 0 && tnum == 1){
            firstlos[i] <- firstlos[i] + 1
            if(resqual == 1){
                lossqual[i] <- lossqual[i] + 1
            }
        }else if(res2 == 3 && tnum == 4){
            firstlos[i] <- firstlos[i] + 1
            if(resqual == 1){
                lossqual[i] <- lossqual[i] + 1
            }
        }else if(res2 == 0 && tnum == 3){
            firstlos[i] <- firstlos[i] + 1
            if(resqual == 1){
                lossqual[i] <- lossqual[i] + 1
            }
        }
    }
    
    ##DONT NEED KO rounds for this!!
    
    rm(list = "sim1", "finpts", "cmatrix", "goalscr", "goaldif", "finalpos", "fint3", "fint3ids", "gsqual")
    gc()
}

# RENAME RData file to be saved before running the program!! ####

save(onlyqual,firstdraw,drawqual,firstwin,winqual,firstlos,lossqual, file = "./old-qualres1.RData")



