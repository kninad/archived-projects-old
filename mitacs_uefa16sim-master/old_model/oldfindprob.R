allmat <- read.csv("./matchups.csv") #all matches
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