clear all;
clc;
close all;

%% Parameter Setting
globalBits = [16,32,64,128]; 
datasets = {'nus_wide_data_hashing'};                       

N = 5000;

% create a masking agent
% total - 182577
mask = 1:182577;

kernelSamps = [500, 500];                            % sampling size for kernel logistic regression
fprintf('The number of kernel samples [%d] \r', kernelSamps(1));            

dtN = length(datasets);
recallLevelStep = 0.05;

%% SePH
for di = 1 : dtN
    clearvars -except globalBits datasets dtN di recallLevelStep fid Model kernelSamps N mask;
    load(['datasets/', datasets{di}, '.mat']);
    
    qs_set_no = 4000; % for CMCQ paper
%     qs_set_no = 1866; % 1% for the seph paper
    
    % For NUS Wide dataset
    % consider 4000(2% randomly sampled pairs) as the query set and the
    % rest as the training set
    t = randperm(size(labels,1));    
    I_te = image_feat(t(1:qs_set_no),:);
    T_te = text_feat(t(1:qs_set_no),:);
    L_te = labels(t(1:qs_set_no),:);
    I_tr = image_feat(t(qs_set_no+1:end),:);
    T_tr = text_feat(t(qs_set_no+1:end),:);
    L_tr = labels(t(qs_set_no+1:end),:);
    sampleInds = 1:size(I_tr,1);
    clear t image_feat text_feat labels
    
    sampleInds = sampleInds(1:N);    
    
    v = 2;
    viewsName = {'Image', 'Text'};
    
    RetrXs = cell(1, v);                            % Retrieval Set
    RetrXs{1} = I_tr(mask,:);
    RetrXs{2} = T_tr(mask,:); 
    L_tr = L_tr(mask,:);
    
    queryXs = cell(1, v);                           % Query Set
    queryXs{1} = I_te;
    queryXs{2} = T_te;    
    clear I_tr T_tr I_te T_te;
    
    % Feature Pretreatment
    for i = 1 : v
        meanV = mean(RetrXs{i}, 1);
        RetrXs{i} = bsxfun(@minus, RetrXs{i}, meanV);
        queryXs{i} = bsxfun(@minus, queryXs{i}, meanV);
    end    

    trainNum = length(sampleInds);                  % Training Set
    trainXs = cell(1, v);
    trainXs{1} = RetrXs{1}(sampleInds, :);
    trainXs{2} = RetrXs{2}(sampleInds, :);
    
    % Calculation of P for supervised learning (normalized cosine similarity)
    tr_labels = L_tr(sampleInds, :);
    
    % for the nuswide datasets
    T1 = tr_labels; T1 = normr(T1); T2 = tr_labels; T2 = normr(T2);
    P = T1*(T2.');    
    
    % Training & Testing
    bitN = length(globalBits);
    bits = globalBits;
    
    queryNum = size(L_te, 1);
    
    runtimes = 1;                                                  % 10 runs
    mAPs = zeros(bitN, v, runtimes, 2);
    trainMAPs = zeros(bitN, runtimes);
    
    for bi = 1 : bitN        
        bit = bits(bi);        

        for ri = 1 : runtimes
            
            %%
            tic
            % Generate the Hash Codes
            % To start the process to stop random initialization
            a = -1; b = 1;
            A0 = (b-a)*rand(size(P,1),bit,'double') + a;
            A0 = sign(A0);
            B0 = (b-a)*rand(size(P,2),bit,'double') + a;
            B0 = sign(B0);
            
            % matrix update hash code learning stage
            [A,B] = generate_hash_codes8_matrix_update(P,size(P,1),size(P,2),bit,A0,B0,1);
            
%             % Evaluating the Quality of Learnt Hash Codes for Training Set
%             trEv = trainEval2(tr_labels, A, B);
%             fprintf('Runtime %d, Manifold Evaluation MAP [%.4f]\r', ri, trEv);            
%             trainMAPs(bi, ri) = trEv;
            
            
            %%
            % RBF Kernel
            z = trainXs{1} * trainXs{1}';
            z = repmat(diag(z), 1, trainNum)  + repmat(diag(z)', trainNum, 1) - 2 * z;
            k1 = {};
            k1.type = 0;
            k1.param = mean(z(:));                                  %  $\sigma^2$ for RBF kernel in image view

            z = trainXs{2} * trainXs{2}';
            z = repmat(diag(z), 1, trainNum)  + repmat(diag(z)', trainNum, 1) - 2 * z;
            k2 = {};
            k2.type = 0;
            k2.param = mean(z(:));                                  %  $\sigma^2$ for RBF kernel in text view    
            
            % p(c_k=1) and p(c_k=-1)
            learntP1 = [sum(A == 1, 1) / size(A, 1); sum(A == -1, 1) / size(A, 1);];
            learntP2 = [sum(B == 1, 1) / size(B, 1); sum(B == -1, 1) / size(B, 1);];
            
            % Kernel Logistic Regression (KLR)£¬Developed by Mark Schimidt     
            for si = 1 : 1
                kernelSampleNum = kernelSamps(di);
                if si == 1 && kernelSampleNum > trainNum
                    break;
                elseif si == 2 && kernelSampleNum > trainNum / 2
                    break;
                end

                sampleType = 'Random';
                if si == 1
                    % Random Sampling for Learning KLR
                    kernelSamples = sort(randperm(trainNum, kernelSampleNum));
                    kernelXs{1} = trainXs{1}(kernelSamples, :);
                    kernelXs{2} = trainXs{2}(kernelSamples, :);
                else
                    sampleType = 'Kmeans';
                    % Kmeans Sampling for Learning KLR
                    opts = statset('Display', 'off', 'MaxIter', 100);
                    [INX, C] = kmeans(trainXs{1}, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
                    kernelXs{1} = C;

                    [INX, C] = kmeans(trainXs{2}, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
                    kernelXs{2} = C;
                end

                % Kernel Matrices
                K01 = kernelMatrix(kernelXs{1}, kernelXs{1}, k1);
                K02 = kernelMatrix(kernelXs{2}, kernelXs{2}, k2);
                trainK1 = kernelMatrix(trainXs{1}, kernelXs{1}, k1);
                trainK2 = kernelMatrix(trainXs{2}, kernelXs{2}, k2);
                RetrK1 = kernelMatrix(RetrXs{1}, kernelXs{1}, k1);
                RetrK2 = kernelMatrix(RetrXs{2}, kernelXs{2}, k2);
                queryK1 = kernelMatrix(queryXs{1}, kernelXs{1}, k1);
                queryK2 = kernelMatrix(queryXs{2}, kernelXs{2}, k2);        

                % Hash Codes for Retrieval Set and Query Set
                B1 = zeros(size(L_tr, 1), bit);                             % Unique Hash Codes for Both Views of Retrieval Set
                B21 = zeros(queryNum, bit);                                 % Hash Codes for Image View of Query Set
                B22 = zeros(queryNum, bit);                                 % Hash Codes for Text View of Query Set
                
                
                options.Display = 'final';        
                options.MaxIter = 500;
                C = 0.01;                                                   % Weight for Regularization. 1e-2 is Good Enough.
                
                h = waitbar(0,'Please wait...');
                
                % KLR for Each Bit
                for b = 1 : bit
                    tH = A(:, b);
                    ppos1 = 1 / learntP1(1, b);               % 1/p(c_k=1)
                    pneg1 = 1 / learntP1(2, b);               % 1/p(c_k=-1)
                    ppos1(isinf(ppos1)|isnan(ppos1)) = 1;
                    ppos1(isinf(pneg1)|isnan(pneg1)) = 1;
                    
                    % View 1 (Image View)
                    funObj = @(u)LogisticLoss(u, trainK1, tH);
                    w = minFunc(@penalizedKernelL2, zeros(size(K01, 1),1), options, K01, funObj, C);  
                    B21(:, b) = sign(queryK1 * w);
                    z11 = 1 ./ (1 + exp(-RetrK1 * w));                                     % P(pos | V_1)
                    z10 = 1 - z11;                                                         % P(neg | V_1)
                    
                    tH = B(:, b);
                    ppos2 = 1 / learntP2(1, b);               % 1/p(c_k=1)
                    pneg2 = 1 / learntP2(2, b);               % 1/p(c_k=-1)
                    ppos2(isinf(ppos2)|isnan(ppos2)) = 1;
                    ppos2(isinf(pneg2)|isnan(pneg2)) = 1;
                    
                    % View 2 (Text View)
                    funObj = @(u)LogisticLoss(u, trainK2, tH);
                    w = minFunc(@penalizedKernelL2, zeros(size(K02, 1),1), options, K02, funObj, C); 
                    B22(:, b) = sign(queryK2 * w);
                    z21 = 1 ./ (1 + exp(-RetrK2 * w));                                     % P(pos | V_2)     
                    z20 = 1 - z21;                    

                    wt = 0.5;
                    B1(:, b) = sign(wt*(ppos1*z11 - pneg1*z10)+(1-wt)*(ppos2*z21 - pneg2*z20));
                    
                    waitbar(b/bit,h);
                end
                
                close(h);
                
                B1 = bitCompact(sign(B1) >= 0);
                B21 = bitCompact(sign(B21) >= 0);
                B22 = bitCompact(sign(B22) >= 0);                
                
                % Evaluation
                vi = 1;
                hammingM = 1-double(HammingDist(B21, B1))';
                mAPValue = map_at_50(hammingM,L_tr,L_te);
                mAPs(bi, vi, ri, si) = mAPValue;
                fprintf('%s Bit %d Runtime %d Sampling Type [%s] Sampling Num [%d], %s query %s: MAP [%.6f]\r', ...,
                    datasets{di}, bit, ri, sampleType, kernelSampleNum, viewsName{1}, viewsName{2}, mAPValue);
                
                clear hammingM mAPValue
                
                vi = 2;
                hammingM = 1-double(HammingDist(B22, B1))';
                mAPValue = map_at_50(hammingM,L_tr,L_te);                
                mAPs(bi, vi, ri, si) = mAPValue;
                fprintf('%s Bit %d Runtime %d Sampling Type [%s] Sampling Num [%d], %s query %s: MAP [%.6f]\r', ...,
                    datasets{di}, bit, ri, sampleType, kernelSampleNum, viewsName{2}, viewsName{1}, mAPValue);                
                
                clear hammingM mAPValue
            toc
            end            
        end
        [mean(mAPs(bi,1,:,1)) mean(mAPs(bi,2,:,1))]
        [mean(mAPs(bi,1,:,2)) mean(mAPs(bi,2,:,2))]
    end
end