# function to calculate the AUCR and AUC
calcAUPR <- function(obsLabel, predProb, fpr.stop=0.5) {
  unique_labels <- unique(obsLabel)
	# cat("unique labels:", unique_labels, "\n")
	# flush.console()
  if (length(unique_labels) != 2) stop("The first argument 'obsLabel' should be two classes!\n")
	# calculate AUC using ROCR
	# library('ROCR')
  pred <- ROCR::prediction(predProb, obsLabel)
  if(fpr.stop == "cost"){
    perf <- ROCR::performance(pred, "cost")
    fpr.stop <- pred@cutoffs[[1]][which.min(perf@y.values[[1]])]
    perf <- ROCR::performance(pred, "auc", fpr.stop=fpr.stop)
  }else{
    perf <- ROCR::performance(pred, "auc", fpr.stop=fpr.stop)
  }
  
  auc <- as.numeric(perf@y.values)
	
  # Calculate AUPR using ROCR
  perf <- ROCR::performance(pred, 'rec', 'prec', fpr.stop=fpr.stop)
  Precision <- unlist(perf@x.values)
  Recall <- unlist(perf@y.values)

  # library('MESS')
  aupr_spline <- try(MESS::auc(Recall, Precision, type = 'spline'), silent = TRUE)

	# Save the result
	statRes <- matrix(0, nrow = 1, ncol = 2)
	colnames(statRes) <- c("auc", "aupr")

  if (class(aupr_spline) == 'try-error') {
    # library('Bolstad2')
    # uses Simpson's rule: numerical integrating, solve the area
    aupr_simpson <- Bolstad2::sintegral(Recall, Precision)$int
		statRes[, "auc"] <- auc
		statRes[, "aupr"] <- aupr_simpson
  } else {
    statRes[, "auc"] <- auc
		statRes[, "aupr"] <- aupr_spline
  }
	return(statRes)
}






