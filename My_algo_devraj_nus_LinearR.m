clear all; 
clc; 
close all;
addpath(genpath('markSchmidt/'));
warning off;

%% Parameter Setting
globalBits = [16,32,64,128];
datasets = {'nus_wide_data_hashing'};

N = 5000;

% create a masking agent
% total - 182577
mask = 1:182577;

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
    
    runtimes = 10;                                                  % 10 runs
    mAPs = zeros(bitN, v, runtimes);
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
            % p(c_k=1) and p(c_k=-1)
            learntP1 = [sum(A == 1, 1) / size(A, 1); sum(A == -1, 1) / size(A, 1);];
            learntP2 = [sum(B == 1, 1) / size(B, 1); sum(B == -1, 1) / size(B, 1);];
            
            % Learning linear ridge regression via 5-fold cross-validation,
            % Estimating the corresponding parameters of the Gaussian Distributions w.r.t -1 and 1
            [w1, gaussian1] = AdaptiveTrainLinearRidgeRegression_CV(trainXs{1}, A, 2);
            [w2, gaussian2] = AdaptiveTrainLinearRidgeRegression_CV(trainXs{2}, B, 2);            
            
            % Hash Codes for Retrieval Set and Query Set
            B1 = zeros(size(L_tr, 1), bit);             % Unique Hash Codes for Both Views of Retrieval Set
            B21 = zeros(size(L_te, 1), bit);            % Hash Codes for Image View of Query Set
            B22 = zeros(size(L_te, 1), bit);            % Hash Codes for Text View of Query Set
            
            options = {};
            options.Display = 'final';
            C = 0.01;                                   % Weight for Regularization. We Found that 1e-2 is Good Enough.
            
            h = waitbar(0,'Please wait...');
            
            % KLR for Each Bit
            for b = 1 : bit
                tH = A(:, b);
                ppos1 = 1 / learntP1(1, b);               % 1/p(c_k=1)
                pneg1 = 1 / learntP1(2, b);               % 1/p(c_k=-1)
                ppos1(isinf(ppos1)|isnan(ppos1)) = 1;
                ppos1(isinf(pneg1)|isnan(pneg1)) = 1;
                
                % View 1 (Image View)
                w = w1(:, b);
                B21(:, b) = sign(queryXs{1} * w);
                z = RetrXs{1} * w;
                zz1 = exp(-((z - gaussian1(1, 1, b)) .^ 2) / (2*(gaussian1(2, 1, b)^2))) / gaussian1(2, 1, b);
                zz2 = exp(-((z - gaussian1(1, 2, b)) .^ 2) / (2*(gaussian1(2, 2, b)^2))) / gaussian1(2, 2, b);
                z11 = zz1 ./ (zz1 + zz2);                 % P(pos | V_1)
                z10 = 1 - z11;                            % P(neg | V_1)
                
                tH = B(:, b);
                ppos2 = 1 / learntP2(1, b);               % 1/p(c_k=1)
                pneg2 = 1 / learntP2(2, b);               % 1/p(c_k=-1)
                ppos2(isinf(ppos2)|isnan(ppos2)) = 1;
                ppos2(isinf(pneg2)|isnan(pneg2)) = 1;
                
                % View 2 (Text View)
                w = w2(:, b);
                B22(:, b) = sign(queryXs{2} * w);
                z = RetrXs{2} * w;
                zz1 = exp(-((z - gaussian2(1, 1, b)) .^ 2) / (2*(gaussian2(2, 1, b)^2))) / gaussian2(2, 1, b);
                zz2 = exp(-((z - gaussian2(1, 2, b)) .^ 2) / (2*(gaussian2(2, 2, b)^2))) / gaussian2(2, 2, b);
                z21 = zz1 ./ (zz1 + zz2);                 % P(pos | V_1)
                z20 = 1 - z21;                            % P(neg | V_1)                

                wt = 0.5;
                B1(:, b) = sign(wt*(ppos1*z11 - pneg1*z10)+(1-wt)*(ppos2*z21 - pneg2*z20));
                
                % computation here %
                waitbar(b/bit,h);
            end
            close(h);
            
            B1 = bitCompact(sign(B1) >= 0);
            B21 = bitCompact(sign(B21) >= 0);
            B22 = bitCompact(sign(B22) >= 0);
            
            clear learntP1 learntP2 A B f w1 w2 gaussian1 gaussian2
            clear C options ppos1 ppos2 pneg1 pneg2 tH zz1 zz2 z11 z10 z21 z20
            
            % Evaluation
            vi = 1;
            hammingM = 1-double(HammingDist(B21, B1))';
            mAPValue = map_at_50(hammingM,L_tr,L_te);
            mAPs(bi, vi, ri) = mAPValue;
            fprintf('%s Bit %d Runtime %d, %s query %s: MAP [%.6f]\r', ...,
                datasets{di}, bit, ri, viewsName{1}, viewsName{2}, mAPValue);
            
            clear hammingM mAPValue
            
            vi = 2;
            hammingM = 1-double(HammingDist(B22, B1))';
            mAPValue = map_at_50(hammingM,L_tr,L_te);
            mAPs(bi, vi, ri) = mAPValue;
            fprintf('%s Bit %d Runtime %d, %s query %s: MAP [%.6f]\r', ...,
                datasets{di}, bit, ri, viewsName{2}, viewsName{1}, mAPValue);
            
            clear hammingM mAPValue
            
            toc
        end
        [mean(mAPs(bi,1,:)) mean(mAPs(bi,2,:))]
    end    
end