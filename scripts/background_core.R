# Local per-array normexp processing through the installed Bioconductor limma.
# No cohort-fitting or between-array normalization occurs in this script.
options(warn=1)
.libPaths(c(normalizePath("data/local_runtime/R-library"), .libPaths()))
library(limma)
logdir <- "context/core-reprocessing/v1"
jobs <- read.delim(file.path(logdir,"background-jobs.tsv"),check.names=FALSE)
stopifnot(identical(jobs$study,c("GSE102971","GSE8917","GSE65292")))
params <- list(); checks <- list(); fits <- list(); allwarnings <- list()
for (i in seq_len(nrow(jobs))) {
  job <- jobs[i,]
  x <- matrix(readBin(job$input,"double",n=job$rows*job$columns,size=8,endian="little"),nrow=job$rows)
  stopifnot(ncol(x)==job$columns,all(colSums(is.finite(x))>100))
  result <- matrix(NA_real_,nrow(x),ncol(x)); localfits <- list()
  for (j in seq_len(ncol(x))) {
    warnings <- character()
    fit <- withCallingHandlers(normexp.fit(x[,j],method="saddle"),warning=function(w) {warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
    stopifnot(length(fit$par)==3,all(is.finite(fit$par)),fit$convergence==0)
    y <- withCallingHandlers(normexp.signal(fit$par,x[,j]),warning=function(w) {warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
    stopifnot(identical(is.na(y),is.na(x[,j])),all(is.finite(y[!is.na(y)])),all(y[!is.na(y)]>0))
    result[,j] <- y; localfits[[j]] <- fit
    params[[length(params)+1]] <- data.frame(study=job$study,column=j,mu=fit$par[1],sigma=exp(fit$par[2]),
        alpha=exp(fit$par[3]),convergence=fit$convergence,m2loglik=fit$m2loglik,minimum=min(y,na.rm=TRUE),warnings=paste(warnings,collapse=" | "))
    if(length(warnings)) allwarnings[[paste(job$study,j)]] <- warnings
  }
  # Public backgroundCorrect API provides an independent composition check.
  api <- backgroundCorrect(x,method="normexp",normexp.method="saddle",offset=0,verbose=FALSE)
  diff <- max(abs(api-result),na.rm=TRUE)
  reversed <- backgroundCorrect(x[,rev(seq_len(ncol(x))),drop=FALSE],method="normexp",normexp.method="saddle",offset=0,verbose=FALSE)
  orderdiff <- max(abs(reversed-result[,rev(seq_len(ncol(x))),drop=FALSE]),na.rm=TRUE)
  singlediff <- max(abs(backgroundCorrect(x[,1,drop=FALSE],method="normexp",normexp.method="saddle",offset=0,verbose=FALSE)-result[,1]),na.rm=TRUE)
  stopifnot(diff==0,orderdiff==0,singlediff==0)
  writeBin(as.double(result),job$output,size=8,endian="little")
  fits[[job$study]] <- localfits
  checks[[i]] <- data.frame(study=job$study,api_max_difference=diff,reversed_batch_max_difference=orderdiff,
                            single_sample_max_difference=singlediff,nonpositive_corrected=sum(result<=0,na.rm=TRUE),
                            missing_cells=sum(is.na(result)))
  cat("Completed",job$study,ncol(x),"arrays; minimum corrected",min(result,na.rm=TRUE),"\n")
}
write.table(do.call(rbind,params),file.path(logdir,"background-parameters.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
write.table(do.call(rbind,checks),file.path(logdir,"background-checks.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
saveRDS(fits,"data/processed_pilot/core-reprocessing-v1/background-fits.rds")
capture.output(sessionInfo(),file=file.path(logdir,"R-sessionInfo.txt"))
writeLines(capture.output(str(allwarnings)),file.path(logdir,"background-warnings.txt"))
stopifnot(length(allwarnings)==0)
cat("COMPLETE: all 145 per-array fits converged; API/order/single checks passed\n")
