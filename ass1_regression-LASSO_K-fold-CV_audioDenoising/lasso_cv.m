function [wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t,X,lambdavec,K)
% [wopt,lambdaopt,VMSE,EMSE] = lasso_cv(t,X,lambdavec)
% Calculates the LASSO solution problem and trains the hyperparameter using
% cross-validation.
%
%   Output: 
%   wopt        - mx1 LASSO estimate for optimal lambda
%   lambdaopt   - optimal lambda value
%   MSEval      - vector of validation MSE values for lambdas in grid
%   MSEest      - vector of estimation MSE values for lambdas in grid
%
%   inputs: 
%   y           - nx1 data column vector
%   X           - nxm regression matrix
%   lambdavec   - vector grid of possible hyperparameters
%   K           - number of folds

[N,M] = size(X);
Nlam = length(lambdavec);
%disp(['Nlam size: ', num2str(length(Nlam))])

% Preallocate
SEval = zeros(K,Nlam);
SEest = zeros(K,Nlam);


% cross-validation indexing
% ... Select random indices for validation and estimation
%randomind = cvpartition(N,'KFold', K );
randomind = crossvalind('kfold', N, K);    % XXXXXXX

% Index start when moving through the folds
location = 0;
% How many samples per fold
Nval = floor(N/K);
% How many samples to skip when moving to the next fold.
hop = Nval; 


for kfold = 1:K
    % ... Select validation indices
    %valind = randomind.test(kfold); 
    %valind = find(randomind == 1);
    %valind = find(randomind == kfold);
    valind = find(randomind == kfold);
    
    %disp(['valind size: ', num2str(size(valind))])
    
    % ... Select estimation indices
    %estind = randomind.training(kfold);
    %estind = find(randomind ~= 1);
    estind = find(randomind ~= kfold);
    %estind = ~valind;
    
    % assert empty intersection between valind and estind
    assert(isempty(intersect(valind,estind)),...
        'There are overlapping indices in valind and estind!');
    
    % Initialize estimate for warm-starting.
    wold = zeros(M,1); 
    
    for klam = 1:Nlam
        % ... Calculate LASSO estimate on estimation indices 
        % for the current lambda-value.
        currlam = lambdavec(klam);
        what = lasso_ccd(t(estind),X(estind,:),currlam);   % XXXXXX
        %disp(['what size: ', num2str(size(what))])
        %disp(['X size: ', num2str(size(X))])
        %size(norm((t(valind(klam))-X(valind(klam))*what(klam)),2))
        
        % ... Calculate validation error for this estimate
        %SEval(kfold,klam) = ...
         %   valind(klam)^(-1)*(norm((t(valind(klam))-...
           %                             X(valind(klam))*what(klam)),2))^2;
%         SEval(kfold,klam) = ...
%             valind(kfold)^(-1)*(sqrt(sum(abs(t(valind(kfold))- ...
%                                         X(valind(kfold))*what(kfold))^2)))^2;
        SEval(kfold,klam) = ...
            Nval^(-1)*sum((t(valind)-X(valind,:)*what).^2); 
        
        SEest(kfold,klam) = ...
            (N-Nval)^(-1)*sum((t(estind)-X(estind,:)*what).^2); 


        %SEval(kfold,klam) = ...
         %  (Nval)^(-1)*(sqrt(sum(abs(t(valind)- ...
          %                         X(valind,:)*what)^2)))^2;
         
        % ... Calculate estimation error for this estimate
        %SEest(kfold,klam) = ...
         %   estind(klam)^(-1)*(sqrt(sum(abs(t(estind(klam))- ...
          %                         X(estind(klam))*what(klam))^2)))^2;
%         SEest(kfold,klam) = ...
%             estind(kfold)^(-1)*(sqrt(sum(abs(t(estind(kfold))- ...
%                                      X(estind(kfold))*what(kfold))^2)))^2; 
%         
        % Set current estimate as old estimate for next lambda-value.
        wold = what; 
        % Display current fold and lambda-index.
        disp(['Fold: ' num2str(kfold) ', lambda-index: ' num2str(klam)]) 
        
    end
    
    % Hop to location for next fold. 
    location = location + hop; 
end

% Calculate MSE_val as mean of validation error over the folds.
MSEval = mean(SEval,1); 
% Calculate MSE_est as mean of estimation error over the folds.
MSEest = mean(SEest,1); 
% ... Select optimal lambda 
[mini,minIndex] = min(MSEval);  % XXXXX RMSEval XXXXXXX
lambdaopt = lambdavec(minIndex);
disp(['lambdaopt: ', num2str(lambdaopt)])

RMSEval = sqrt(MSEval);
disp(['RMSEval: ', num2str(RMSEval)])
RMSEest = sqrt(MSEest);
disp(['RMSEest: ', num2str(RMSEest)])

% ... Calculate LASSO estimate for selected lambda using all data.
wopt = lasso_ccd(t,X,lambdaopt); 


end

