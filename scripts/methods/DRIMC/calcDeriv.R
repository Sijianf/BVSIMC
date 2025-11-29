
calcDeriv <- function(
  cc = 5,
  inMat,
  U,
  V,
  Sd,
  St,
  lamU,
  lamV,
  isGradU = TRUE) {
  
  # INPUT
  # cc: default 5
  # inMat: input interaction matrix
  # U: latent matrix for rows
  # V: latent matrix for cols
  # Sd: similarity for drugs
  # St: similarity for target
  # lamU: lambda for U
  # lamV: lambda for V
  # isGradU: TRUE or FALSE
    
  # OUTPUT
  # gradient of U or V
  
  
  Y <- inMat
  cY <- cc * Y
  
  StV <- St %*% V
  StVt <- t(StV)
  SdU <- Sd %*% U
  
  A <- SdU %*% StVt
  
  M <- 1 + cY - Y
  
  sigmoidRes <- sigmoid(A)
  P <- M * sigmoidRes
  
  if (isGradU) {
    Dt <- t(Sd)  
    gradU <-
      Dt %*% (cY - P) %*% StV - lamU * U 
    return(gradU)
  } else {
    Tt <- t(St)
    cYt <- t(cY)
    Pt <- t(P)
    gradV <-
      Tt %*% (cYt - Pt) %*% SdU - lamV * V
    return(gradV)
  }
}
