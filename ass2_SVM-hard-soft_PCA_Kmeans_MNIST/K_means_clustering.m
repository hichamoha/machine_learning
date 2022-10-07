function [y,C] = K_means_clustering(X,K)

% returns cluster assignments y and cluster centroids C given 
% an input of data X the number of clusters K.

% Input:    X   DxN matrix of input data
%           K   Number of clusters
%
% Output:   y   Nx1 vector of cluster assignments
%           C   DxK matrix of cluster centroids

[D,N] = size(X)

intermax = 50;
conv_tol = 1e-6;

% Initialize the K centroids
C = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
y = zeros(N,1);
Cold = C;

for kiter = 1:intermax

    % Step 1: Assign examples to one of the K clusters
    y = step_assign_cluster(X, Cold);
    %y = [];
%     for i = 1:N
%         d = fxdist(X(:,i), Cold);
%         Find the closest cluster to put X(i) in
%         [minDist,minIndex] = min(d);    % XXXX what about k index ???
%         y(i) = minIndex;
%     end
    
    % Step 2: Assign new clusters centroids
    [C, Cdist] = step_compute_mean(X, Cold, K, y);
%     C = [];
    %sumXi = 0;
%     for k = 1:K
%         indx = find(y == k);
%         C(:,k) = mean(X(:, indx), 2);
%         %Nk = sum(y == k);
%         %N(k) = N(k) + 1;
%         %sumXi = sumXi + X;
%         %C(:,k) = sum(X(y == k))/Nk;    % XXXXX X(i)
%     end 
    %end
        
    % check convergence
    %if fcdist(C,Cold) < conv_tol
    if Cdist < conv_tol
        return
    end
    Cold = C;
end

end

function d = fxdist(x,C)
%     d = [];
    %for i = 1:N
    for j = 1:size(C,2)
        d(j) = norm(x - C(:,j));
    end
    %end 
end

function d = fcdist(C1,C2)
    %d = [];
    d = norm(C1 - C2);
    if C1 == C2
        d = 0;
    end 
end

function [y] = step_assign_cluster(X, C)
    % Step 1: Assign examples to one of the K clusters
    %y = [];
    for i = 1:size(X, 2)
        d = fxdist(X(:,i), C);
        % Find the closest cluster to put X(i) in
        [minDist,minIndex] = min(d);    % XXXX what about k index ???
        y(i) = minIndex;
    end

end

function [C, Cdist] = step_compute_mean(X, Cold, K, y)
    % Step 2: Assign new clusters centroids

    %sumXi = 0;
    for k = 1:K
        indx = find(y == k);
        C(:,k) = mean(X(:, indx), 2);
        %Nk = sum(y == k);
        %N(k) = N(k) + 1;
        %sumXi = sumXi + X;
        %C(:,k) = sum(X(y == k))/Nk;    % XXXXX X(i)
        %Cdist = fcdist(C,Cold);
    end 
    Cdist = fcdist(C,Cold);

end