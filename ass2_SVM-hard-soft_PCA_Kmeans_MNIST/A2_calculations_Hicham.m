

% Data
xi = [-2 -1 1 2];
yi = [1 -1 -1 1];

%% Task 1
% Compute the kernel matrix K using the data from the table.
K = zeros(length(xi), length(yi));
for i = 1:length(xi)
    for j = 1:length(xi)
        K(i,j) = phi(xi(i))'* phi(xi(j));
    end
end
disp('By using the feature map \phi(x), we get the kernel matrix K: ')
K

%% Task2
% Solve the maximization problem in (5) for α numerically, 
% using the data in (2)
sumDenominator = 0;
for i = 1:length(xi)
    for j = 1:length(yi)
        %sum = sum + yi(i)*yi(j)*(xi(i)*xi(j) + (xi(i)^2)*(xi(j)^2));
        sumDenominator = sumDenominator + yi(i)*yi(j)*K(i,j);
    end
end
alpha = 4 /(sumDenominator);
disp(['Solving the maximization problem in (5), we get \alpha: ',...
                                                 num2str(alpha)])
                                             
%% Task 3
xs = xi(4);
ys = yi(4);
sumalphayk = 0;
for j = 1:length(xi)
    %sumalphayk = sumalphayk + alpha*yi(j)*phi(xi(j))'*phi(xs);
    sumalphayk = sumalphayk + alpha*yi(j)*K(j,4);
end
b = (1/ys) - sumalphayk

%format rat


