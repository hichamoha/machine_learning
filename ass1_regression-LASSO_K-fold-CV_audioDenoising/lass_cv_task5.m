 
% download data.
%clearvars, 
clear
close all, 
clc
load('A1_data')

lambdamax = max(abs(X'*t))
lambda_grid = exp(linspace(log(0.1), log(lambdamax), 10));
%lambda_grid = linspace(0.1, 20, 10);
[wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t, X, lambda_grid, 5);

%% Produce plots illustrating RMSEval(lambda) and RMSEest(lambda)
% Plot the original N = 50 data points with disconnected
% markers (no connecting lines) vs the time axis given by n.
% Plot the original data t
figure(1);
hold on;
plot(lambda_grid, RMSEval,'b'); 
%semilogx(lambda_grid, RMSEval,'b');
%scatter(n, t, 'b*')

xlabel('\lambda_i');
ylabel('RMSA');
title('The root mean squared error vs \lambda_i',...,
      'FontWeight','bold', 'Color','b')

% Overlay in the same plot RMSEest
plot(lambda_grid, RMSEest,'m');  
%semilogx(lambda_grid, RMSEest,'b');

hold on;

% Also add a dashed vertical line at the location of the lambdaopt
% of the data, vs. an interpolated time axis.
%plot(lambdaopt,'g-')
line([lambdaopt lambdaopt], get(gca, 'ylim'), ... 
     'LineStyle','- -', 'Color', 'g')

legend('RMSEval','RMSEest', 'location of optimal \lambda');

%% Produce reconstruction plots
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
      ; ['\lambda = ' num2str(lambdaopt)]},'FontWeight','bold', 'Color','b')

% Overlay these with N = 50 reconstructed data points, i.e.,
% y = X*what(??), also using disconnected markers
y = X*wopt;
%display('y size is: ')
%size(y)
%
plot(n, y,'ro'); 
hold on;

% add a solid line without markers for an interpolated reconstruction 
% of the data, vs. an interpolated time axis.
plot(ninterp, Xinterp*wopt, 'Color', 'g')
legend('Original data',...
       'Reconstructed data', 'Interpolated reconstruction');


 