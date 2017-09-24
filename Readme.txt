This submission is the extension of the implementation of the paper 
"Generalized Semantic Preserving Hashing for N-Label Cross-Modal Retrieval"
in CVPR 2017 (which has been submitted to TIP for acceptance).

**************************************************************
PLEASE READ THIS FIRST
THERE ARE SEVERAL CHANGES AS COMPARED TO THE CVPR SUBMISSION
[1] in CVPR we used only the element-wise updates
[2] here we used column updates and matrix updates (which is 
very fast) thus enabling us to use larger number of training data
pairs.
[3] ***Extra experiments for the MIRFLICKR dataset have been provided.***
[4] ***Also provided are the results using the linear regression and
logistic regression (with a kernel pre-processing stage)
**************************************************************

***********************************************************
The implementations provided are for the Wiki and NUS-Wide 
dataset. You can replicate the results of the paper using that.

[1] Please unzip the markSchmidt.zip file to get started.
[2] Put the data in .mat file in the folder datasets.

Kindly look into the following program to better understand
essence of the algorithm
(1) generate_hash_codes8_matrix_update.m

This implementation uses the post-unification startegy.

Please change the number of iterations (as you seem fit) or run
the algorithm until convergence.
In case you need the data kindly contact me separately.
***********************************************************

***********************************************************
In case you are needed to use this code for unpaired scenarios
you just need to remove the post-unification strategy.
***********************************************************

***********************************************************
I am unable to give the codes for the normal cross-modal operations
due to license issues. Please download the data from the places 
as instructed in the paper. For the CNN features you need to download 
the dataset and extract the features yourself.
I had used matconvnet at http://www.vlfeat.org/matconvnet/ to do that

However to get you started, I have provided the following codes to you
so that the evaluation can be done quickly
(1) retrieval.m - code to compute the PrecisionatK and NDCGatK
***********************************************************
