% PCA Algorithm in Matlab - cmu example	

clear all
% generate data
% R = mvnrnd(MU,SIGMA,N) returns a N-by-D matrix R of random vectors
% chosen from the multivariate normal distribution with 1-by-D mean
% vector MU, and D-by-D covariance matrix SIGMA.
Data = mvnrnd([5, 5],[1 1.5; 1.5 3], 100);

figure(1); 
plot(Data(:,1), Data(:,2), '+');
title('Visualisation of data generated randomly')

% center the data
for i = 1:size(Data,1)
  Data(i, :) = Data(i, :) - mean(Data);
end

 %covariance matrix
DataCov = cov(Data);
[PC, variances, explained] = pcacov(DataCov); %eigen

% plot principal components
figure(2); 
clf; 
hold on;
plot(Data(:,1), Data(:,2), '+b');

plot(PC(1,1)*[-5 5], PC(2,1)*[-5 5], '-r')
plot(PC(1,2)*[-5 5], PC(2,2)*[-5 5], '-b'); 

title('plot of the zero-mean data with the principal components')

hold off

% project down to 1 dimension
PcaPos = Data * PC(:, 1);
