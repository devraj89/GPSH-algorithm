function [k_traindata,k_testdata] = apply_kernel(traindata,testdata,n_anchors)
Ntrain = size(traindata,1);

% % % % [~,anchor] = kmeans(traindata,n_anchors);
anchor = traindata(randperm(Ntrain, n_anchors),:);

Dis = EuDist2(traindata,anchor,0);
sigma = mean(min(Dis,[],2).^0.5);
clear Dis
feaTrain = exp(-sqdist(traindata,anchor)/(2*sigma*sigma));
feaTest = exp(-sqdist(testdata,anchor)/(2*sigma*sigma)); 
m = mean(feaTrain);
k_traindata = bsxfun(@minus, feaTrain, m);
k_testdata = bsxfun(@minus, feaTest, m);


% % % % rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);
% % % % feaTrain =  [rbfKernel(traindata,anchor)];
% % % % feaTest =   [rbfKernel(testdata,anchor)];
% % % % m = mean(feaTrain);
% % % % k_traindata = bsxfun(@minus, feaTrain, m);
% % % % k_testdata = bsxfun(@minus, feaTest, m);