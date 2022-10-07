function [L1, L2, newLabels ] = K_means_classifier(X, C, labels, newX)

% Step 1: Assign examples to one of the K clusters
    %y = [];
for i = 1:size(X,2)
    d = fxdist(X(:,i), C);
    % Find the closest cluster to put X(i) in
    [minDist,minIndex] = min(d);    
    y(i) = minIndex;
end

% Step 2: Assign labels L1 and L2 to the 2 cluster centroids 

% mode() compute the most frequent values of  
L1 = mode(labels(y == 1));
[count1, lbl] = hist(labels(y == 1), unique(labels))
L2 = mode(labels(y == 2));
[count1, lbl] = hist(labels(y == 2), unique(labels))



for i = 1:size(newX,2)
    d = fxdist(newX(:,i), C);
    % Find the closest cluster to put X(i) in
    [minDist,minIndex] = min(d);    
    if minIndex == 1
        newLabels(i) = L1;
    else
        newLabels(i) = L2;
    end
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