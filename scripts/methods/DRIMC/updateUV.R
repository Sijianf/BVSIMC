updateUV <- function(cc = 5,
                     inMat,
                     Sd,
                     St,
                     lamU,
                     lamV,
                     numLat = 50,
                     initMethod = "useNorm",
                     thisSeed = 123,
                     iterpara,
                     maxIter = 100) {
  
  # INPUT
  # cc:
  # inMat:
  # Sd:
  # St:
  # lamU:
  # lamV:
  # numLat:
  # initMethod:
  # thisSeed:  
  # maxIter:
  
  # OUTPUT:
  # a list with two elements: U and V
  
  deltaLL_save <-c()
  LL_save <-c()
  numRow <- nrow(inMat)
  numCol <- ncol(inMat)
  
  if (initMethod == "useNorm") {
    U <- matrix(NA, nrow = numRow, ncol = numLat)
    U <- apply(U, 2, function(x) {
      rnorm(x, mean = 0, sd = 1)
    })
    U <- sqrt(1 / numLat) * U
    V <- matrix(NA, nrow = numCol, ncol = numLat)
    V <- apply(V, 2, function(x) {
      rnorm(x, mean = 0, sd = 1)
    })
    V <- sqrt(1 / numLat) * V
  } else if (initMethod == "useSeed") {
    set.seed(thisSeed)
    U <- matrix(NA, nrow = numRow, ncol = numLat)
    U <- apply(U, 2, function(x) {
      rnorm(x, mean = 0, sd = 1)
    })
    U <- sqrt(1 / numLat) * U
    V <- matrix(NA, nrow = numCol, ncol = numLat)
    V <- apply(V, 2, function(x) {
      rnorm(x, mean = 0, sd = 1)
    })
    V <- sqrt(1 / numLat) * V
  } else {
    stop("initMethod should be one of {useNorm, useSeed}\n")  
  }
  
  sumGradU <- matrix(0, nrow = numRow, ncol = numLat)
  sumGradV <- matrix(0, nrow = numCol, ncol = numLat)

  # last log-likelihood
  lastLog <- calcLogLik(
      cc = cc,
      inMat = inMat,
      U = U,
      V = V,
      Sd = Sd,
      St = St,
      lamU = lamU,
      lamV = lamV)
  
  currDeltaLL <- 1000
  # main loop
  for (i in 1:maxIter) {
    # gradU
    gradU <- calcDeriv(
      cc = cc,
      inMat = inMat,
      U = U,
      V = V,
      Sd = Sd,
      St = St,
      lamU = lamU,
      lamV = lamV,     
      isGradU = TRUE)
    sumGradU <- sumGradU + (gradU ^ 2)
    stepSize <- iterpara / sqrt(sumGradU)
    U <- U + stepSize * gradU
    
    # gradV
    gradV <- calcDeriv(
      cc = cc,
      inMat = inMat,
      U = U,
      V = V,
      Sd = Sd,
      St = St,
      lamU = lamU,
      lamV = lamV,
      isGradU = FALSE
    )
    sumGradV <- sumGradV + (gradV ^ 2)
    stepSize <- iterpara / sqrt(sumGradV)
    V <- V + stepSize * gradV
    
    currLog <- calcLogLik(
      cc = cc,
      inMat = inMat,
      U = U,
      V = V,
      Sd = Sd,
      St = St,
      lamU = lamU,
      lamV = lamV)
    
    # delta log-likelihood
    deltaLog <- (currLog - lastLog) / abs(lastLog)
    
    # stop earlier
    if (abs(deltaLog) < 1e-5) {
      break
    }
    
    if ((i > 50) & (deltaLog > currDeltaLL)) {
      break
    }
    
    currDeltaLL <- deltaLog
    lastLog <- currLog
    
    deltaLL_save <-c(deltaLL_save,currDeltaLL)
    LL_save <-c(LL_save,lastLog)
  }
  
  UV <- list(U = U, V = V, deltaLL = deltaLL_save, LL = LL_save)
  return(UV)
}
