calcLogLik <- function(
  cc = 5,
  inMat,
  U,
  V,
  Sd,
  St,
  lamU,
  lamV) {
  
  # INPUT
  # cc: default 5
  # inMat: input interaction matrix
  # U: latent matrix for rows
  # V: latent matrix for cols
  # Sd: similarity for drugs
  # St: similarity for target
  # lamU: lambda for U
  # lamV: lambda for V
  
  # OUTPUT
  # a scalar of log-likelihood
    
  
  Y <- inMat
  cY <- cc * Y
  Ut <- t(U)
  Vt <- t(V)
  StV <- St %*% V
  StVt <- t(StV)
  SdU <- Sd %*% U
  
  A <- SdU %*% StVt

  # 2017-07-18, numerical stability
  log1pexpRes <- log1pexp(A)
  
  LL <- sum((1 + cY - Y) * log1pexpRes) - sum(cY * A) +
    0.5 * lamU * (base::norm(U, "F") ^ 2) + 0.5 * lamV * (base::norm(V, "F") ^ 2)
  return(LL)
}
