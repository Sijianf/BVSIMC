calcPredScore <- function(U,
                          V,
                          simDrug,
                          simTarget,
                          knownDrugIndex,
                          knownTargetIndex,
                          testIndexRow,
                          testIndexCol,
                          K = 5,
                          testLabel,
                          fpr.stop=0.5) {
  # INPUT
  # U: row latent matrix
  # V: col latent matrix
  # simDrug: similarity matrix for drug, but diagonal elements are zeros
  # simTarget: similarity matrix for target, but diagonal elements are zeros
  # testIndexRow: row index for test set
  # testIndexCol: col index for test set
  # K: number of nearest neighbor for prediction
  # testLabel: labels for the test set
  
  # OUTPUT 
  # a list of AUC and AUPR
  
  if (K < 0) {
    stop("K MUST be '>=' 0! \n")
  }
  
  if (K > 0) {
    ## cat("with K smoothing! \n")
    ## for drug
    indexTestD <- unique(testIndexRow)
    testD <- U[indexTestD, ]
    testD <- cbind(indexTestD, testD)
    numTest <- length(indexTestD)
    numColTestD <- ncol(testD)
    simDrugKnown <- simDrug[, knownDrugIndex]
    numDrugKnown <- length(knownDrugIndex)
    
    for (i in 1:numTest) {
      indexCurr <- indexTestD[i]
      isNewDrug <- !(indexCurr %in% knownDrugIndex)
      if (isNewDrug) {
        simDrugNew <- simDrugKnown[indexCurr, ] # vector
        indexRank <- rank(simDrugNew) # vector
        indexNeig <- which(indexRank > (numDrugKnown - K))
        simCurr <- simDrugNew[indexNeig] # vector
        # index for U
        index4U <- knownDrugIndex[indexNeig]
        U_Known <- U[index4U, , drop = FALSE] # force to matrix
        # vec %*% matrix => matrix
        if (sum(simCurr) != 0){
           testD[i, 2:numColTestD] <- (simCurr %*% U_Known) / sum(simCurr)
        }
      }
    }
    
    Unew <- U
    Unew[indexTestD, ] <- testD[, -1]
    
    ## for target
    # unique index for test target
    indexTestT <- unique(testIndexCol)
    testT <- V[indexTestT, ]
    # add first column as labels
    testT <- cbind(indexTestT, testT) # 1st column is unique test label
    # number of unique test set
    numTest <- length(indexTestT)
    # number of column for testT
    numColTestT <- ncol(testT)
    # known similarity matrix for targets
    simTargetKnown <- simTarget[, knownTargetIndex]
    # number of known targets
    numTargetKnown <- length(knownTargetIndex)
    
    for (i in 1:numTest) {
      indexCurr <- indexTestT[i]
      isNewTarget <- !(indexCurr %in% knownTargetIndex)
      if (isNewTarget) {
        simTargetNew <- simTargetKnown[indexCurr, ] # vector
        indexRank <- rank(simTargetNew) # vector
        # selected neighbor index with top K neighbor
        indexNeig <- which(indexRank > (numTargetKnown - K))
        # get similarity value of K
        simCurr <- simTargetNew[indexNeig] # vector
        # index for V
        index4V <- knownTargetIndex[indexNeig]
        V_Known <- V[index4V, , drop = FALSE] # force to matrix
        # vec %*% matrix => matrix
        if (sum(simCurr) != 0){
           testT[i, 2:numColTestT] <- (simCurr %*% V_Known) / sum(simCurr)
        }
      }
    }

    Vnew <- V
    Vnew[indexTestT, ] <- testT[, -1]
    
    StV <- simTarget %*% Vnew
    StVt <- t(StV)
    SdU <- simDrug %*% Unew
    val <- SdU %*% StVt
    testSetIndex <- cbind(testIndexRow, testIndexCol)
    val <- val[testSetIndex]
    
    # score from val
    # score <- exp(val) / (1 + exp(val))
    score <- val
    for (h in 1:length(val)){
      if (val[h] >= 0){
        z = exp(-val[h])
        score[h] = 1 / (1 + z)
      }
      else{
        z = exp(val[h])
        score[h] = z / (1 + z)
      }
    }
    result <- calcAUPR(testLabel, score, fpr.stop=fpr.stop)
  } else {  # K = 0 condition
    # cat("without K smoothing! \n")
    # flush.console()
    Vt <- t(V)
    UVt <- U %*% Vt
    StV <- simTarget %*% V
    StVt <- t(StV)
    SdU <- simDrug %*% U
    val <- SdU %*% StVt 
    testSetIndex <- cbind(testIndexRow, testIndexCol)
    val <- val[testSetIndex]
    # score
    # score <- exp(val) / (1 + exp(val))
    # score <- sigmoid(val)
    score <- val
    for (h in 1:length(val)){
      if (val[h] >= 0){
        z = exp(-val[h])
        score[h] = 1 / (1 + z)
      }
      else{
        z = exp(val[h])
        score[h] = z / (1 + z)
      }
    }
    result <- calcAUPR(testLabel, score, fpr.stop=fpr.stop)
  }
  return(result)
}
