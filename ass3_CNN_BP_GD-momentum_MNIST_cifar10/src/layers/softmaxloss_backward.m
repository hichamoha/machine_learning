function dldx = softmaxloss_backward(x, labels)
    % Inputs:
    %    x - Features. See the reshape command below. It is reshaped as for
    %        the fully connected layer.
    %    labels - It is a vector with the correct labels. For
    %        instance if we have a batch of two where the first example is
    %        class 4 and the second example is class 7, labels is [4 7].
    %
    % Outputs:
    %    dldx - Partial derivative of L with respect to x. Remember that in
    %           the forward pass you average over the batch elements.
    labels = vec(labels);
    sz = size(x);
    batch = sz(end);
    features = prod(sz(1:end-1));

    % suitable for matrix multiplication
    x = reshape(x, [features, batch]);
    % for numerical stability. Convince yourself that the result is the same.
    x = bsxfun(@minus, x, min(x, [], 1));

    %error('Implement this!');
    %y = exp(x) ./ sum(exp(x));
    sumExpx = sum(exp(x), 1);
    %sumExpx = bsxfun(@plus, exp(x), 1:batch )  % XXXXX
    %y = exp(x) ./ sumExpx; % here can use bsxfun(@rdivide, exp(x), sumExpx)
    y = bsxfun(@rdivide, exp(x), sumExpx);
    y(sub2ind(sz, labels', 1:batch)) = y(sub2ind(sz, labels', 1:batch)) - 1;
    dldx = y;
    dldx = dldx / batch;
    
    
    
end
