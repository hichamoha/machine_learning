function [Wopt,lambdaopt,RMSEval,RMSEest] = ...
                                     multiframe_lasso_cv(T,X,lambdavec,K)
% [wopt,lambdaopt,VMSE,EMSE] = multiframe_lasso_cv(T,X,lambdavec,n)
% Calculates the LASSO solution for all frames and trains the
% hyperparameter using cross-validation.
%
%   Output:
%   Wopt        - mxnframes LASSO estimate for optimal lambda
%   lambdaopt   - optimal lambda value
%   VMSE        - vector of validation MSE values for lambdas in grid
%   EMSE        - vector of estimation MSE values for lambdas in grid
%
%   inputs:
%   T           - NNx1 data column vector
%   X           - NxM regression matrix
%   lambdavec   - vector grid of possible hyperparameters
%   K           - number of folds

% Define some sizes
NN = length(T);
[N,M] = size(X);
Nlam = length(lambdavec);

% Set indexing parameters for moving through the frames.
framehop = N;
idx = (1:N)';
framelocation = 0;
Nframes = 0;
while framelocation + N <= NN
    Nframes = Nframes + 1; 
    framelocation = framelocation + framehop;
end % Calculate number of frames.

% Preallocate
Wopt = zeros(M,Nframes);
SEval = zeros(K,Nlam);
SEest = zeros(K,Nlam);

% Set indexing parameter for the cross-validation indexing
Nval = floor(N/K);
cvhop = Nval;

% ... Select random indices for picking out validation and estimation indices. 
randomind = crossvalind('kfold', N, K);
    
framelocation = 0;
for kframe = 1:Nframes % First loop over frames
    
    cvlocation = 0;
    
    for kfold = 1:K % Then loop over the folds
        
        % ... Select validation indices
        valind = find(randomind == kfold);
        % ... Select estimation indices
        estind = find(randomind ~= kfold); 
        
        % assert empty intersection between valind and estind
        assert(isempty(intersect(valind,estind)), ...
            'There are overlapping indices in valind and estind!'); 
    
        % Set data in this frame
        t = T(framelocation + idx); 
        % Initialize old weights for warm-starting.
        wold = zeros(M,1); 
        
        % Finally loop over the lambda grid
        for klam = 1:Nlam  
            
            % ... Calculate LASSO estimate at current frame, fold, and lambda
            currlam = lambdavec(klam);
            what = lasso_ccd(t(estind),X(estind,:),currlam);
            
            %  ...Add validation error at current frame, fold and lambda to 
            % the validation error for this fold and lambda, summing the  
            % error over the frames
%             SEval(kfold,klam) = ...
%             valind(kfold)^(-1)*(sqrt(sum(abs(t(valind(kfold))- ...
%                                         X(valind(kfold))*what(kfold))^2)))^2;
            SEval(kfold,klam) = ...
                           Nval^(-1)*sum((t(valind)-X(valind,:)*what).^2); 
%             
%             %  ... Add estimation error at current frame, fold and lambda 
%             % to the estimation error for this fold and lambda, summing 
%             % the error over the frames
%             SEest(kfold,klam) = ...
%             estind(kfold)^(-1)*(sqrt(sum(abs(t(estind(kfold))- ...
%                                      X(estind(kfold))*what(kfold))^2)))^2;
       
            SEest(kfold,klam) = ...
                        (N-Nval)^(-1)*sum((t(estind)-X(estind,:)*what).^2); 


            
            % Set current LASSO estimate as estimate for warm-starting.
            wold = what; 
            
             % Display progress through frames, folds and lambda-indices.
            disp(['Frame: ' num2str(kframe) ...
                  ', Fold: ' num2str(kfold) ...
                  ', Hyperparam: ' num2str(klam)])
        end
        
        % Hop to location for next fold.
        cvlocation = cvlocation+cvhop;
    end
    
    % Hop to location for next frame.
    framelocation = framelocation + framehop; 
    
end


% Average validation error across folds
MSEval = mean(SEval,1);
% Average estimation error across folds
MSEest = mean(SEest,1); 
% ... Assign optimal lambda 
[mini,minIndex] = min(MSEval);  % XXXXX RMSEval XXXXXXX
lambdaopt = lambdavec(minIndex);

% Move through frames and calculate LASSO estimates using both estimation
% and validation data, store in Wopt.
framelocation = 0;
for kframe = 1:Nframes
    t = T(framelocation + idx);
    Wopt(:,kframe) = NaN;
    framelocation = framelocation + framehop;
end

RMSEval = sqrt(MSEval);
RMSEest = sqrt(MSEest);

end

