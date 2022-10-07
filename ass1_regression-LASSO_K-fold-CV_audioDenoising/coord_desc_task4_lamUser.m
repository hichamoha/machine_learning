% download data.
%clearvars,
clear
close all, clc
load('A1_data')

%% Coordinate descent minimizer
% The regression matrix is given by X in
% it consists of 500 candidate sine and cosine pairs
%[N,M] = size(X);
%XTX = X'*X;
l = 0.2;
%wold = zeros(M, 1);
%lasso_ccd(t,X,lambda,wold)
what = lasso_ccd(t,X,l);
%size(what)

%% Produce reconstruction plots
% Plot the original N = 50 data points with disconnected
% markers (no connecting lines) vs the time axis given by n.
% Plot the original data t
figure;
hold on;
plot(n, t,'b*'); 
%scatter(n, t, 'b*')
xlabel('time axis given by n');
ylabel('the original N = 50 data points');
title('\fontsize{12} Reconstruction plots of the noisy data t',...,
      'FontWeight','bold', 'Color','b')

% Overlay these with N = 50 reconstructed data points, i.e.,
% y = X*what(λ), also using disconnected markers
y = X*what;
%display('y size is: ')
%size(y)
%
plot(n, y,'ro'); 
hold on;

% add a solid line without markers for an interpolated reconstruction 
% of the data, vs. an interpolated time axis.
plot(ninterp, Xinterp*what, 'Color', 'g')
legend('\fontsize{8} Original data',...
       '\fontsize{8} Reconstructed data',...
       '\fontsize{8} Interpolated reconstruction');

