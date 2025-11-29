

rm(list = ls())
#setwd('D:\\Program\\submit')

## current data set name
# db <- "lrssl"
# db <- "predict"
db <- "c"

switch (db,
        predict = {
          cat("predict data\n")
          flush.console()
          sd1 <- read.table("predict_simmat_dc_chemical.txt")
          sd1 <- as.matrix(sd1)
          sd2 <- read.table("predict_simmat_dc_domain.txt")
          sd2 <- as.matrix(sd2)
          sd3 <- read.table("predict_simmat_dc_go.txt")
          sd3 <- as.matrix(sd3)
          st <- read.table("predict_simmat_dg.txt")
          st <- as.matrix(st)
          Y <- read.table("predict_admat_dgc.txt")
          Y <- as.matrix(Y)
          Y <- t(Y)
        },
        lrssl = {
          cat("lrssl data\n")
          flush.console()
          sd1 <- read.table("lrssl_simmat_dc_chemical.txt", sep = "\t", strip.white = T, header = T, row.names = 1)
          sd1 <- as.matrix(sd1)
          sd2 <- read.table("lrssl_simmat_dc_domain.txt", sep = "\t", strip.white = T, header = T, row.names = 1)
          sd2 <- as.matrix(sd2)
          sd3 <- read.table("lrssl_simmat_dc_go.txt", sep = "\t", strip.white = T, header = T, row.names = 1)
          sd3 <- as.matrix(sd3)
          st <- read.table("lrssl_simmat_dg.txt", sep = "\t", strip.white = T, header = T, row.names = 1)
          st <- as.matrix(st)
          Y <- read.table("lrssl_admat_dgc.txt", sep = "\t", strip.white = T, header = T, row.names = 1)
          Y <- as.matrix(Y)
        },
        c = {
          cat("c data\n")
          flush.console()
          sd1 <- read.table("c_simmat_dc_chemical.txt")
          sd1 <- as.matrix(sd1)
          sd2 <- read.table("c_simmat_dc_domain.txt")
          sd2 <- as.matrix(sd2)
          sd3 <- read.table("c_simmat_dc_go.txt")
          sd3 <- as.matrix(sd3)
          st <- read.table("c_simmat_dg.txt")
          st <- as.matrix(st)
          Y <- read.table("c_admat_dgc.txt")
          Y <- as.matrix(Y)
          Y <- t(Y)
        },
        stop("db should be one of the follows: 
             {predict, lrssl, c}\n")
        )

## load required packages
pkgs <- c("matrixcalc", "data.table", "Rcpp", "ROCR", "Bolstad2", "MESS")
rPkgs <- lapply(pkgs, require, character.only = TRUE)

## source required R files
rSourceNames <- c("doCrossValidationByPairwise.R",
                  "doCrossValidationByRow.R",
                  "constrNeig.R", 
                  "inferZeros.R",
                  "calcLogLik.R",
                  "calcDeriv.R",
                  "updateUV.R",
                  "calcPredScore.R",
                  "calcAUPR.R"
)
rSN <- lapply(rSourceNames, source, verbose = FALSE)

## sourceCPP required C++ files
cppSourceNames <- c("fastKF2.cpp", "fastKF4.cpp", "fastKgipMat.cpp", "log1pexp.cpp", "sigmoid.cpp")
cppSN <- lapply(cppSourceNames, sourceCpp, verbose = FALSE)

## do cross-validation
kfold <- 10

numSplit <- 5

## split training and test sets
# cvMethod <- "cvByRow"
cvMethod <- "cvByPairwise"
seeds <- c(7771, 8367, 22, 1812, 4659)

###############################
if (cvMethod == "cvByPairwise") {
  savedFolds <- doCrossValidationByPairwise(Y, kfold = kfold, numSplit = numSplit, seeds = seeds)
} else if (cvMethod == "cvByRow") {
  savedFolds <- doCrossValidationByRow(Y, kfold = kfold, numSplit = numSplit, seeds = seeds)
} else {
  stop("Please specify the cv method!\n")
}

## hyper-parameters

numLat <- 200 #dimensionality of the subspace
lamU <- 2 #regularization parameter 
lamV <- 2 #regularization parameter 

K1 <- 20 #K-nearest neighbors for inferring association profiles of new drugs or diseases
iterpara <- 0.125 #learning rate
cc <- 10 #confidence level
K2 <- 20 #K-nearest neighbors for drug/disease feature selection 
K3 <- 20 #smooth the latent factors for new drugs or diseases
AUPRVec <- vector(length = kfold)
AUCVec <- vector(length = kfold)
finalResult <- matrix(NA, nrow = numSplit, ncol = 2)
colnames(finalResult) <- c("AUPR", "AUC")
for (i in 1:numSplit) {
  for (j in 1:kfold) {
    cat("numSplit:",
        i,
        "/",
        numSplit,
        ";",
        "kfold:",
        j,
        "/",
        kfold,
        "\n")
    flush.console()
    Y <- savedFolds[[i]][[j]][[7]]
    Yr <- inferZeros(Y, sd1, K = K1)
    Yc <- inferZeros(t(Y), st, K = K1)
    KgipD <- fastKgipMat(Yr, 1)
    KgipT <- fastKgipMat(Yc, 1)
    nNeig <- 3
    nIter <- 5
    sdcomb <- fastKF4(KgipD, sd1, sd2, sd3, nNeig, nIter)
    stcomb <- fastKF2(KgipT, st, nNeig, nIter)
    lap <- constrNeig(sdcomb, stcomb, K = K2)
    simD <- lap$simD
    simT <- lap$simT
    ## use AdaGrid to update U and V
    UV <- updateUV(
      cc = cc,
      inMat = Y,
      Sd = simD,
      St = simT,
      lamU = lamU,
      lamV = lamV,
      numLat = numLat,
      initMethod = "useSeed",
      thisSeed = 123,
      iterpara = iterpara,
      maxIter = 100
    )
    
    U <- UV$U
    V <- UV$V
    
    knownDrugIndex <- savedFolds[[i]][[j]][[5]]
    knownTargetIndex <- savedFolds[[i]][[j]][[6]]
    testIndexRow = savedFolds[[i]][[j]][[3]]
    testIndexCol = savedFolds[[i]][[j]][[4]]
    testLabel = savedFolds[[i]][[j]][[1]]
    ## result
    result <- calcPredScore(
      U = U,
      V = V,
      simDrug = simD,
      simTarget = simT,
      knownDrugIndex = knownDrugIndex,
      knownTargetIndex = knownTargetIndex,
      testIndexRow = testIndexRow,
      testIndexCol = testIndexCol,
      K = K3,
      testLabel = testLabel
    )
    AUPRVec[j] <- result[1, "aupr"]
    AUCVec[j] <- result[1, "auc"]
  }
  AUPR <- mean(AUPRVec)
  AUC <- mean(AUCVec)
  finalResult[i, "AUPR"] <- AUPR
  finalResult[i, "AUC"] <- AUC
}

## print the result
cat(
  "\n======================\n\n",
  "db is: ",
  db,
  "\n",
  ## hyper-parameters
  "numLat = ",
  numLat,
  "\n",
  "cc = ",
  cc,
  "\n",
  "lamU = ",
  lamU,
  "\n",
  "lamV = ",
  lamV,
  "\n",
  "K1 = ",
  K1,
  "\n",
  "K2 = ",
  K2,
  "\n",
  "K3 = ",
  K3,
  "\n",
  "iterpara = ",
  iterpara,
  "\n",
  "cvMethod = ",
  cvMethod,
  "\n",
  "\n=====================\n"
)
cat(numSplit, "trails", kfold, "-fold CV", "\n")
cat("DRIMC: default parameters \n")
cat("aupr:", mean(finalResult[, "AUPR"]), "+/-", sd(finalResult[, "AUPR"]), "\n")
cat("auc:", mean(finalResult[, "AUC"]), "+/-", sd(finalResult[, "AUC"]), "\n")
