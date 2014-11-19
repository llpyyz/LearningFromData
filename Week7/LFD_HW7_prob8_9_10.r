####################
#Learning from Data
#HW, Week 7
#Problem 8, 9
#8) ans: 20-30%
#9) ans: 35-40%
#10) ans: 3.59 SV's on avg
####################

####################################################
#The following packages are assumed to be installed:
#--LowRankQP
#--iterators
#--random
####################################################
library(LowRankQP)
library(random)


#generates random uniform pt in [a,b] X [c,d]
gen_pt <- function(a = -1, b = 1, c = -1, d = 1){
    c(runif(1 , a , b), runif(1 , c , d))
}

#calc slope or return Inf
get_slope <- function(pt1, pt2){

    slope <- Inf
    if(pt1[1] != pt2[1]){
	    slope <- (pt1[2] - pt2[2])/(pt1[1] - pt2[1])
	}
	slope
}

#returns a data set as a matrix with numpts rows and numfeatures cols
gen_data <- function(numpts, numfeatures = 2){
    mat <- matrix(0, numpts, numfeatures)
	for(i in 1:nrow(mat)){
	    for(j in 1:ncol(mat)){
		    data_pt <- gen_pt()
		    mat[i, j] = data_pt[j]
		}
	}
	mat
}

#classify data
#input: 
#--weight_vec, a vector of weights
#--data_set, a N by d matrix of N data pts with d features each
#
#output: N by 1 matrix with +/-1 classifications
perceptron_classify <- function(weight_vec, data_set){
	2 * ((cbind(matrix(1 , nrow(data_set) , 1), data_set) %*% weight_vec) > 0) - 1
}

perceptron_learning_algo <- function(train_set, train_classify){
	
	#initially all pts misclassified
	misclassified_data <- matrix(T, nrow(train_set) , 1)
	max_iters <- 25
	curr_iter <- 1
	pla_weights <- matrix(0, 1 + ncol(train_set), 1) #init to all 0's
	
	#while still misclassified pts (and have not exceeded max iters)
	#-randomly pick one, adjust weights, adjust list of misclassified
	while(sum(misclassified_data) > 0 && curr_iter <= max_iters){
		
	    #get idx of currently misclassified pt
		idx_vec <- c()
		for(i in 1 : length(misclassified_data)){
		    if(misclassified_data[i]){
			    idx_vec <- c(idx_vec, i)
			}    
		}
		
		#random misclassified pt
		curr_pt_idx <- idx_vec[sample(1:length(idx_vec), 1, replace = T)] 
		curr_pt_sign <- train_classify[curr_pt_idx , 1]
		
		#update weights
		pla_weights <- pla_weights + curr_pt_sign * matrix(c(1, train_set[curr_pt_idx, ]), 1 + ncol(train_set), 1) 
				
		#classify as per new weights
		curr_classifications <- perceptron_classify(pla_weights, train_set) 
		
		#identify misclassified pts
		misclassified_data <- curr_classifications != train_classify 
		
		curr_iter <- curr_iter + 1
	}
	
	pla_weights
}

#Use quadratic programming to calculuate Lagrange multipliers
#then calc weights for SVM
my_svm <- function(train_set, train_classify){
    #Do LowRankQP
	train_classify_rep <- matrix(c(rep(train_classify,ncol(train_set))),nrow(train_set),ncol(train_set))
	Vmat <- (train_set * train_classify_rep) %*% t(train_set * train_classify_rep)
	dvec <- matrix(-1,1,nrow(train_set))
	Amat <- train_classify
	bvec <- 0
	uvec <- matrix(1000, nrow(train_classify), 1)
	svm_res <- LowRankQP(Vmat, dvec, t(Amat), bvec, uvec, method = "CHOL", verbose = F)
	
	#calc svm weights
	svm_wts <- colSums(matrix(rep(svm_res$alpha * train_classify, ncol(train_set)), nrow = nrow(train_set)) * train_set)
	print("")
	print("SVs:")
	print(svm_res$alpha)
	print("")
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

num_runs <- 500
svm_better_than_pla <- 0
overall_sv_count <- 0

for(i in 1 : num_runs){

    #####
	#PLA#
	#####
	
    ##############
    #Create target
	##############
	
    #2 pts, slope
	x1 <- gen_pt()
	x2 <- gen_pt()
	target_slope <- get_slope(x1, x2)
	target_weight_vec <- matrix(c(target_slope * x1[1] - x1[2], -target_slope, 1), 1 + length(x1) , 1)
	
	#get, classify training data
	train_set_size <- 100
	train_set <- gen_data(train_set_size)
	train_classify <- perceptron_classify(target_weight_vec, train_set)
	
	#if all pts in training set on one side, gen/classify new set
	while(abs(sum(train_classify)) == train_set_size){
	    train_set <- gen_data(train_set_size)
	    train_classify <- perceptron_classify(target_weight_vec, train_set)
	}
	
	#learn weights with PLA and SVM
    pla_wts <- perceptron_learning_algo(train_set, train_classify)
	res <- my_svm(train_set, train_classify)
	svm_wts <- res[[1]]
	overall_sv_count <-  overall_sv_count + res[[2]]
	
	####################
	#Test set generation
	####################
	
	#gen and classify large test set
	test_set_size <- 10000
	test_set <- gen_data(test_set_size)
	test_classify <- perceptron_classify(target_weight_vec, test_set)
	
	############################
	#Compare PLA and SVM results
	############################
	
	#find percentage of pts in test where pla and target disagree
	test_classify_pla <- perceptron_classify(pla_wts, test_set) #classify by pla weights
	pla_pct_disagree <- sum(test_classify != test_classify_pla) / length(test_set_size)
	
	#find percentage of pts in test where svm and target disagree
	test_classify_svm <- perceptron_classify(svm_wts, test_set) #classify by pla weights
	svm_pct_disagree <- sum(test_classify != test_classify_svm) / length(test_set_size)
	
	#track whether svm has lower % disagreement than pla
	if(svm_pct_disagree < pla_pct_disagree){
	    svm_better_than_pla <- svm_better_than_pla + 1
	}
}

pct_svm_better_than_pla <- svm_better_than_pla / num_runs
print(paste("in", as.character(num_runs), "runs:"))
print("pct of time svm beats pla:")
print(pct_svm_better_than_pla)
print("mean number of SVs:")
print(overall_sv_count / num_runs)