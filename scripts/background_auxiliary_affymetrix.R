# Explicit HT PM CEL identity and per-array RMA background; no cohort fit.
.libPaths(c("data/local_runtime/R-library",.libPaths()))
suppressPackageStartupMessages(library(affy))
suppressPackageStartupMessages(library(hthgu133pluspmcdf))
options(warn=1)
logdir <- "context/auxiliary-processing/v1"
outdir <- "data/processed_pilot/auxiliary-raw-v1"
jobs <- read.delim(file.path(logdir,"affymetrix-jobs.tsv"),check.names=FALSE)
stopifnot(nrow(jobs)==412, !anyDuplicated(jobs$gsm))
first <- ReadAffy(filenames=jobs$path[1],cdfname="hthgu133pluspm",verbose=FALSE)
indices <- indexProbes(first,which="pm")
ids <- names(indices); ix <- unlist(indices,use.names=FALSE)
probe_ids <- rep(ids,lengths(indices))
stopifnot(length(ix)==nrow(pm(first)),all(ix>=1),all(ix<=744*744))
raw_first <- affyio::read.celfile(jobs$path[1])
stopifnot(max(abs(raw_first$INTENSITY$MEAN[ix]-as.vector(pm(first))))==0)
write.table(data.frame(pm_position=seq_along(ix)-1,physical_index_1based=ix,probeset_id=probe_ids),
            file.path(logdir,"affymetrix-pm-map.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
con <- file(file.path(outdir,"GSE85570_rma_background.bin"),"wb")
qc <- list(); checks <- list()
for(i in seq_len(nrow(jobs))) {
  cel <- affyio::read.celfile(jobs$path[i])
  stopifnot(identical(cel$HEADER$cdfName,"HT_HG-U133_Plus_PM"),
            identical(as.integer(cel$HEADER[["CEL dimensions"]]),c(744L,744L)),
            length(cel$INTENSITY$MEAN)==744*744)
  # Fail closed for marked CEL spots until a separately validated mask adapter exists.
  stopifnot(nrow(cel$MASKS)==0,nrow(cel$OUTLIERS)==0)
  x <- matrix(cel$INTENSITY$MEAN[ix],ncol=1)
  stopifnot(all(is.finite(x)),all(x>=0))
  y <- preprocessCore::rma.background.correct(x)
  stopifnot(all(is.finite(y)),all(y>0))
  writeBin(as.double(y),con,size=8,endian="little")
  qq <- quantile(log2(y),c(0,.25,.5,.75,1))
  qc[[i]] <- data.frame(gsm=jobs$gsm[i],pm_rows=length(ix),physical_pm=length(unique(ix)),
                       masks=0,outliers=0,nonpositive_raw=sum(x<=0),minimum_log2=qq[1],q25_log2=qq[2],
                       median_log2=qq[3],q75_log2=qq[4],maximum_log2=qq[5])
  if(i==1) {
    api <- pm(bg.correct(first,method="rma"))
    checks[[1]] <- data.frame(check="affy_public_background_api",max_difference=max(abs(api-y)))
    stopifnot(max(abs(api-y))<1e-10)
    yfirst <- y; xfirst <- x
  }
  if(i==2) {
    pair <- preprocessCore::rma.background.correct(cbind(x,xfirst))
    checks[[2]] <- data.frame(check="per_array_batch_and_order",max_difference=max(abs(pair-cbind(y,yfirst))))
    stopifnot(max(abs(pair-cbind(y,yfirst)))==0)
  }
  if(i%%25==0) cat("Corrected",i,"/412 CEL arrays\n")
}
close(con)
write.table(do.call(rbind,qc),file.path(logdir,"affymetrix-raw-qc.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
write.table(do.call(rbind,checks),file.path(logdir,"affymetrix-background-checks.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
capture.output(sessionInfo(),file=file.path(logdir,"R-affymetrix-sessionInfo.txt"))
cat("COMPLETE 412 CEL arrays; background only; no fitted normalization\n")
