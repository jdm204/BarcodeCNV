# Optional fixture regeneration: direnv exec . Rscript --vanilla test/data/infercnv/generate.R
# Numerical oracle: actual InferCNV, NOT a reimplementation of its transforms.
suppressPackageStartupMessages(library(infercnv))
stopifnot(as.character(packageVersion('infercnv')) == '1.28.0')
options(scipen=100)
directory='test/data/infercnv'
# Short chromosomes (1, 3 and 7 genes), zeros, highly influential outliers,
# unequal libraries, and one entirely unexpressed gene.
n=12L; m=6L
counts=outer(seq_len(n),seq_len(m),function(i,j) ((i*19+j*31)%%127)*j)
counts[1,]=c(0,0,6000,100,10,300)
counts[5,]=c(9000,10,2,0,100,20)
counts[n,]=0
reference=c(400,20,50,60,5,100,20,400,30,20,10,0)
chrom=c('chr1',rep('chr2',3),rep('chr3',7),'chr4')
ids=paste0('gene',seq_len(n)); colnames(counts)=paste0('barcode',seq_len(m));rownames(counts)=ids
write.table(data.frame(id=ids,chrom,reference,counts,check.names=FALSE),file.path(directory,'counts.tsv'),sep='\t',quote=FALSE,row.names=FALSE)
settings=list(cutoff=0,min_cells_per_gene=1L,smooth_method='pyramidinal',cluster_by_groups=FALSE,
 analysis_mode='samples',HMM=FALSE,denoise=FALSE,prune_outliers=FALSE,z_score_filter=0,up_to_step=14L,
 num_threads=1L,resume_mode=FALSE,no_plot=TRUE,no_prelim_plot=TRUE,save_rds=FALSE,save_final_rds=FALSE)
for (mode in c('external','self','none')) for (width in c(1L,3L,7L,101L)) {
 ref=if (mode=='self') apply(sweep(counts,2,colSums(counts),'/'),1,median) else reference
 if(mode=='self') ref=ref/sum(ref)*median(colSums(counts))
 x=if(mode=='none') counts else cbind(counts,reference=ref)
 ann=data.frame(group=ifelse(colnames(x)=='reference','reference','observations'),row.names=colnames(x))
 genes=data.frame(chr=chrom,start=seq_len(n)*100,stop=seq_len(n)*100+10,row.names=ids)
 obj=CreateInfercnvObject(x,genes,ann,ref_group_names=if(mode=='none') NULL else 'reference')
 output=tempfile();dir.create(output)
 result=do.call(infercnv::run,c(list(infercnv_obj=obj,out_dir=output,window_length=width),settings))
 expected=matrix(NA_real_,n,m,dimnames=dimnames(counts))
 expected[rownames(result@expr.data),]=log2(result@expr.data[,colnames(counts),drop=FALSE])
 write.table(data.frame(id=ids,expected,check.names=FALSE),file.path(directory,paste0(mode,'_',width,'.tsv')),sep='\t',quote=FALSE,row.names=FALSE)
 unlink(output,recursive=TRUE)
}
writeLines(c('InferCNV 1.28.0 numerical oracle; regenerate with generate.R.',
 'Full preprocessing through step 14; all settings recorded in generate.R.',
 capture.output(sessionInfo())),file.path(directory,'provenance.txt'))
