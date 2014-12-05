####################
#Learning from Data
#Final exam,
#Problem 11
#11) 5 SV's
####################

####################################################
#The following packages are assumed to be installed:
#--LowRankQP
#--iterators
#--random
####################################################
library(LowRankQP)
library(random)

#Use quadratic programming to calculuate Lagrange multipliers
#then calc weights for SVM
my_svm <- function(train_set, train_classify){
    #Do LowRankQP
	train_classify_rep <- matrix(c(rep(train_classify,ncol(train_set))),nrow(train_set),ncol(train_set))
	Vmat <- (train_set * train_classify_rep) %*% t(train_set * train_classify_rep)
	dvec <- matrix(-1,1,nrow(train_set))
	Amat <- train_classify
	bvec <- 0
	ub <- 100000
	uvec <- matrix(ub, nrow(train_classify), 1)
	svm_res <- LowRankQP(Vmat, dvec, t(Amat), bvec, uvec, method = "CHOL", verbose = F)
	
	#calc svm weights
	svm_wts <- colSums(matrix(rep(svm_res$alpha * train_classify, ncol(train_set)), nrow = nrow(train_set)) * train_set)
		
	#find w_0 aka 'b':
	#need a SV so find an alpha > 0
	max_alpha <- svm_res$alpha[1]
	idx <- 1
	
    for(i in 2 : length(svm_res$alpha)){
	    if(svm_res$alpha[i] > max_alpha){
		    max_alpha <- svm_res$alpha[i]
			idx <- i
		}    
	}
	b <- train_classify[idx] - (matrix(svm_wts , 1 , length(svm_wts)) %*% matrix(train_set[idx], ncol(train_set), 1))
	svm_wts <- c(b, svm_wts)
	
	sv_count <- 0
	tol <- 1e-5
	for(i in 1 : length(svm_res$alpha)){
	    if(svm_res$alpha[i] > tol){
		    
			sv_count <- sv_count + 1
		}    
	}
	
	list(svm_wts, sv_count)
}


#################
#main driver code
#################

train_set <- matrix(c(-3,0,0,1,3,3,3,2,-1,3,2,-3,5,5), nrow = 7, ncol = 2)
train_classify <- matrix(c(-1,-1,-1,1,1,1,1),nrow=7)
res <- my_svm(train_set, train_classify)
res
