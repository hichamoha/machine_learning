% download data.
%clearvars, 
clear
close all, 
clc
load('A1_data')

% Coordinate descent minimizer
% The regression matrix is given by X in
% it consists of 500 candidate sine and cosine pairs
[N,M] = size(X);
%XTX = X'*X;
l = 0.1;
%wold = zeros(M, 1);
%lasso_ccd(t,X,lambda,wold)
what = lasso_ccd(t,X,l);
%size(what)
countw=0;
for i = 1:M
    if (what(i) > 1e-7)
        countw = countw + 1;
    end
end
disp(['The number of the non-zero coordinates w is: ', num2str(countw)])    

% Produce reconstruction plots
% Plot the original N = 50 data points with disconnected
% markers (no connecting lines) vs the time axis given by n.
% Plot the original data t
figure(1);
hold on;
plot(n, t,'b*'); 
%scatter(n, t, 'b*')

xlabel('time axis given by n');
ylabel('the original N = 50 data points');
title({'\fontsize{12} Reconstruction plots of the noisy data t',...,
      ; '\lambda = 0.1'},'FontWeight','bold', 'Color','b')

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
legend('Original data',...
       'Reconstructed data', 'Interpolated reconstruction');

%nnz(w)

%% When the hyperparameter lambda = 10
l = 5;
%wold = zeros(M, 1);
%lasso_ccd(t,X,lambda,wold)
what2 = lasso_ccd(t,X,l);
%size(what)
countw2=0;
for i = 1:M
    if (what2(i) > 1e-7)
        countw2 = countw2 + 1;
    end
end
disp(['The number of the non-zero coordinates w is: ', num2str(countw2)])

% Produce reconstruction plots
% Plot the original N = 50 data points with disconnected
% markers (no connecting lines) vs the time axis given by n.
% Plot the original data t
figure(2);
hold on;
plot(n, t,'b*'); 
%scatter(n, t, 'b*')
xlabel('time axis given by n');
ylabel('the original N = 50 data points');
title({'\fontsize{12} Reconstruction plots of the noisy data t',...,
      ; '\lambda = 10'},'FontWeight','bold', 'Color','b')
% Overlay these with N = 50 reconstructed data points, i.e.,
% y = X*what(λ), also using disconnected markers
y = X*what2;
%display('y size is: ')
%size(y)
%
plot(n, y,'ro'); 
hold on;

% add a solid line without markers for an interpolated reconstruction 
% of the data, vs. an interpolated time axis.
plot(ninterp, Xinterp*what2, 'Color', 'g')
legend('Original data',...,
       'Reconstructed data', 'Interpolated reconstruction');

%% When the hyperparameter lambda = user
l = 0.5;
%wold = zeros(M, 1);
%lasso_ccd(t,X,lambda,wold)
what3 = lasso_ccd(t,X,l);
%size(what)
countw3=0;
for i = 1:M
    if (what3(i) > 1e-7)
        countw3 = countw3 + 1;
    end
end
disp(['The number of the non-zero coordinates w is: ', num2str(countw3)])

% Produce reconstruction plots
% Plot the original N = 50 data points with disconnected
% markers (no connecting lines) vs the time axis given by n.
% Plot the original data t
figure(3);
hold on;
plot(n, t,'b*'); 
%scatter(n, t, 'b*')
xlabel('time axis given by n');
ylabel('the original N = 50 data points');
title({'\fontsize{12} Reconstruction plots of the noisy data t',...,
      ; '\lambda = user'},'FontWeight','bold', 'Color','b')
% Overlay these with N = 50 reconstructed data points, i.e.,
% y = X*what(λ), also using disconnected markers
y = X*what3;
%display('y size is: ')
%size(y)
%
plot(n, y,'ro'); 
hold on;

% add a solid line without markers for an interpolated reconstruction 
% of the data, vs. an interpolated time axis.
plot(ninterp, Xinterp*what3, 'Color', 'g')
legend('Original data', ...
       'Reconstructed data', 'Interpolated reconstruction');