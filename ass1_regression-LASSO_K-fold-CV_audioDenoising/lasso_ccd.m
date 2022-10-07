function what = lasso_ccd(t,X,lambda,wold)
% what = lasso_ccd(t,X,lambda,wold)
% Solves the LASSO optimization problem using cyclic coordinate descent.
%
%   Output: 
%    what   - Mx1 LASSO estimate using cyclic coordinate descent algorithm
%
%   inputs: 
%   t       - Nx1 data column vector
%   X       - NxM regression matrix
%   lambda  - 1x1 hyperparameter value
%   (optional)
%   wold    - Mx1 lasso estimate used for warm-starting the solution.

% Check for match between t and X
[N,M] = size(X);
if size(t,1) ~= N
    disp('Sizes in t and X do not match!')
    what = [];
    return
end

% nargin: Number of function input arguments
if nargin < 4
    % set wold to zeros if warm-start is unavailable
    wold = zeros(M,1); 
end

% Optimization variables and preallocation
% number of iterations
Niter = 50; 
% at which intensity all variables should be updated.
updatecycle = 5; 
% what is to be considered equal to zero in support.
zero_tol = lambda;
% set intial w to wold from previous lasso estimate, if available
w = wold; 
% defines the non-zero indices of w
wsup = double(abs(w)>zero_tol); 

% calculate residual and use it instead of y-Xw with proper indexing.
r = t - X*w;

for kiter = 1:Niter
    
    % Snippet below is a common way of speeding up the estimation process. 
    % Use it if you like. Basically, only the non-zero estimates are updated
    % at every iteration. The zero estimates are only updated every
    % updatecycle number of iterations. Use to your liking. 
    % Otherwise use contents of else statement.
    if rem(kiter,updatecycle) && kiter>2
        kind_nonzero = find(wsup);
        randind = randperm(length(kind_nonzero));
        kindvec_random = kind_nonzero(randind);
    else
        kindvec = 1:M;
        kindvec_random = kindvec(randperm(length(kindvec)));
    end
    
    % sweep over coordinates, in randomized order defined by kInd_random
    count = 0;
    for ksweep = 1:length(kindvec_random)
        % Pick out current coordinate to modify.
        kind = kindvec_random(ksweep); 
        
        % ... select current regression vector
        x = X(:,kind);
        %size(x)
        %size(r)
        % ... put impact of old w(kind) back into the residual.
        r = r + x * w(kind);
        % ... update the lasso estimate at coordinate kind
        if (abs(x'*r)>zero_tol)
            w(kind) = (abs(x'*r) - zero_tol)*(sign(x'*r)/(x'*x));
            count = count + 1;
        else
            w(kind) = 0;
        %w(kind) = (abs(x'*r) - zero_tol)*(x'*r)/((x'*x)*abs(x'*r));
        end
        % ... remove impact of newly estimated w(kind) from residual.
        r = r - x*w(kind);
        
        % update whether w(kind) is zero or not.
        wsup(kind) = double(abs(w(kind))>zero_tol); 
   
    end
end

% assign function output.
what = w; 

%if (what > 1e-7)
%    countw = countw + 1;
end

