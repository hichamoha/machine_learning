 
% download data.
%clearvars, 
clear
close all, 
clc
load('A1_data')

%soundsc(Ttrain, fs);

%lambdamax = max(abs(X'*t))
lambda_grid = exp(linspace(log(0.005), log(1), 10));
%lambda_grid = linspace(0.1, 20, 10);
[wopt,lambdaopt,RMSEval,RMSEest] = ... 
    multiframe_lasso_cv(Ttrain, Xaudio, lambda_grid, 5);

%% listen to the datasets
soundsc(Ttrain, fs);

%% Produce plots illustrating RMSEval(lambda) and RMSEest(lambda)
% Plot the original N = 50 data points with disconnected
% markers (no connecting lines) vs the time axis given by n.
% Plot the original data t
figure(1);
hold on;
plot(lambda_grid, RMSEval,'b'); 
%scatter(n, t, 'b*')

xlabel('\lambda_i');
ylabel('RMSA');
title('The root mean squared error vs \lambda_i',...,
      'FontWeight','bold', 'Color','b')

% Overlay in the same plot RMSEest
plot(lambda_grid, RMSEest,'m');  
hold on;

% Also add a dashed vertical line at the location of the lambdaopt
% of the data, vs. an interpolated time axis.
%plot(lambdaopt,'g-')
line([lambdaopt lambdaopt], ...
      get(gca, 'ylim'), 'LineStyle','- -', 'Color','g')

legend('RMSEval','RMSEest', 'location of optimal \lambda');

%% Task 7 Denoising the the test data Ttest
Ttestclean = lasso_denoise(Ttest,Xaudio,lambdaopt);

%% Listen to the clean data and save it
soundsc(Ttestclean, fs)
save('denoised_audio', 'Ttestclean', 'fs')

%% Produce reconstruction plots
% Plot the original N = 50 data points with disconnected
% markers (no connecting lines) vs the time axis given by n.
% Plot the original data t
figure(2);
hold on;
naudio = linspace(0,length(Ttrain),length(Ttrain) );
plot(naudio, Ttrain,'b*'); 
%scatter(n, t, 'b*')

xlabel('time axis given by naudio');
ylabel('the Ttrain data points');
title('\fontsize{12} plot of the noisy data Ttrain',...,
      'FontWeight','bold', 'Color','b')

% Overlay these with reconstructed data points, i.e.,
% y = X*what(λ), also using disconnected markers
y = Xaudio*wopt;
%display('y size is: ')
%size(y)
nyaudio = linspace(0,352,352);
plot(nyaudio, y,'ro'); 
hold on;




 