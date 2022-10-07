%% load data MNIST_01
% where for both sets the images have been stacked column-wise, 
% i.e., x_i = vec(X_i) in R^(28X28) = R^784
clear all
clc
load('A2_data.mat');

%% Preprocessing the data - standardizing
% the first step in PCA is to standardize the data.
% Here, "standardization" means subtracting the sample mean from each 
% observation, then dividing by the sample standard deviation.
% This centers and scales the data.

%calculate the dimension of data matrix
traindata = train_data_01;
%testdata = test_data_01;
[Dtrain,Ntrain] = size(traindata);
%[Dtest,Ntest] = size(testdata);

meantrain = mean(traindata,2);
stdtrain = std(traindata);
%meantest = mean(testdata);
%stdtest = std(testdata);

% Zero-means data: substract the mean from the data
Xtrain = traindata - repmat(meantrain,[1 Ntrain]) ;
% Standarised data matrices must not be used in this assigment wiht PCA
%Xtrain = traindata ... 
 %       - repmat(meantrain,[Dtrain 1]) ./ repmat(stdtrain, [Dtrain 1]);
Ltrain = train_labels_01;

%% Factorization using the singular value decomposition SVD
% with U, V orthogonal matrices, and S a diagonal matrix.
% U contains the eigenvectors of C = XX^T, the covariance matrix
% S contains the square roots of the eigenvalues of C.
[U,S,V] = svd(Xtrain);

% The score matrix T
%T = U * S;
%figure(1)
%scatter(T,T)

%% Compute the the principal components
%{
The dimensionality reduction of X to dimension d is then the projection of 
X onto the d left singular vectors with the largest absolute singular 
values.
%}


% generat the PCA space (PCA scores), projection of X
%PC = U(:,1:2)' * Xtrain;
PC = U' * Xtrain;


%% Display the results in a plot
figure(1)
gscatter(PC(1,:), PC(2,:), Ltrain)
%gscatter(PC(:,1), PC(:,2), Ltrain)

%plot PCA space of the first two PCs: PC1 and PC2
%scatter(PC(1,:), PC(2,:), '.')
%title('Principal Component Analysis')
%plot(PC(:,1), PC(:,2), Ltrain)
%n = linspace(0,length(Xtrain),length(Xtrain) );
%plot(Xtrain, Ltrain, 'r')
%scatter(n, Xtrain, 'r+')
%axis equal
%
xlabel('1st Principal Compontent')
ylabel('2nd Principal Component')
title('\fontsize{12} The MNIST image data set using PCA in 2D',...,
      'FontWeight','bold', 'Color','b')
legend('Images of zeros','Images of ones');

%% Task E2 - K = 2 
[y2,C2] = K_means_clustering(train_data_01, 2);

figure(2)
gscatter(PC(1,:), PC(2,:), y2, 'bg', '+o')
%plot(U(:,1), U(:,2), 'r+')
%axis equal
xlabel('1st Principal Compontent')
ylabel('2nd Principal Component')
title({'Clustering MNIST image dataset using PCA in 2D',...,
      ; 'K = 2'}, 'FontWeight','bold', 'Color','b')
legend('cluster 1','cluster 2');

%% Task E2 - K = 5 
[y5,C5] = K_means_clustering(train_data_01, 5);

figure(3)
gscatter(PC(1,:), PC(2,:), y5, 'bgrmc', '+o*d^' )
%plot(U(:,1), U(:,2), 'r+')
%axis equal
xlabel('1st Principal Compontent')
ylabel('2nd Principal Component')
title({'Clustering MNIST image dataset using PCA in 2D',...,
      ; 'K = 5'}, 'FontWeight','bold', 'Color','b')
legend('cluster 1','cluster 2', 'cluster 3', 'cluster 4', 'cluster 5');

%% Task E3 - K = 2 centroids as images
centroidImages = reshape(C2, 28, 28, 2);

figure(4)
subplot(1, 2, 1)
imshow(centroidImages(:, :, 1))
title('Image of centroid from cluster 1',...,
      'FontWeight','bold', 'Color','b')
subplot(1, 2, 2)
imshow(centroidImages(:, :, 2))
title('Image of centroid from cluster 2',...,
      'FontWeight','bold', 'Color','b')
% figure(44)
% centroidImages = reshape(C2, 28, 28, 2);
% im1 = centroidImages(:, :, 1);
% im2 = centroidImages(:, :, 2);
% imshow([im1, im2])
% title('\fontsize{12} The K centroids as images',...,
%       'FontWeight','bold', 'Color','b')
 
 %% Task E3 - K = 5 centroids as images
centroidImages = reshape(C5, 28, 28, 5);
figure(5)
for im = 1:5
    subplot(1, 5, im)
    imshow(centroidImages(:, :, im))
end

% a = axes;
% t1 = title('Images of centroids from the 5 clusters',...,
%       'FontWeight','bold', 'Color','b');
% a.Visible = 'off'; % set(a,'Visible','off');
% t1.Visible = 'on'; % set(t1,'Visible','on');


%% Task E4 - Train data Evaluation how many missclassifications
[L1, L2, newLabels ] = K_means_classifier(train_data_01, C2,...,
                                        train_labels_01, train_data_01);
%
%nbX_cluster1 = sum(newLabels' == L1)
%nb_zeros_cluster1 = sum(newLabels' == 0)
%nb_ones_cluster1 = sum(newLabels' == 1)

% nbX_cluster2 = sum(newLabels' == L2)
% 
% train_corrects = sum(newLabels' == train_labels_01)
% train_erros = sum(newLabels' ~= train_labels_01)

%% Task E4 - Test data evaluation
[testL1, testL2, testNewLabels] = K_means_classifier(test_data_01, C2,...,
                                        test_labels_01, test_data_01);
% test_corrects = sum(testNewLabels' == test_labels_01)
% test_erros = sum(testNewLabels' ~= test_labels_01)

%% Task E6 - soft-margin SVM classification

[Dtest,Ntest] = size(test_data_01);

% training SVM using Matlab’s built-in routine
svm_model = fitcsvm(train_data_01', train_labels_01);

% use our trained model to classify the examples in the test data.
predictions_train_svm = predict(svm_model, train_data_01');

% Missclassification usning confusion matrix
confusionmat(train_labels_01, predictions_train_svm)

% [L1, L2, newLabels ] = K_means_classifier(train_data_01, C2,...,
%                                         train_labels_01, train_data_01);

% Test data
% training SVM using Matlab’s built-in routine
%svm_model = fitcsvm(test_data_01', test_labels_01);

% use our trained model to classify the examples in the test data.
predictions_test_svm = predict(svm_model, test_data_01');

% Missclassification usning confusion matrix
confusionmat(test_labels_01, predictions_test_svm)

%% Task E7 . Non-linear kernel SVM classifier
% training SVM using Matlab’s built-in routine
Gsvm_model = fitcsvm(train_data_01', train_labels_01, ...,
                     'KernelFunction', 'gaussian');

Gpredictions_train_svm = predict(Gsvm_model, train_data_01');
confusionmat(train_labels_01, Gpredictions_train_svm)

% use our trained model to classify the examples in the test data.
Gpredictions_test_svm = predict(Gsvm_model, test_data_01');

% Missclassification usning confusion matrix
confusionmat(test_labels_01, Gpredictions_test_svm)

% Misclassification for the traing data
% [L1, L2, newLabels ] = K_means_classifier(train_data_01, C2,...,
%                                     train_labels_01, train_data_01); 

%% Task E7 . Non-linear kernel SVM classifier with tuning beta
% training SVM using Matlab’s built-in routine
Gsvm_beta_model = fitcsvm(train_data_01', train_labels_01, ...,
                     'KernelFunction', 'gaussian', 'KernelScale', 5);

% use our trained model to classify the examples in the test data.
GpredictionsBeta_test_svm = predict(Gsvm_beta_model, test_data_01');

% Missclassification usning confusion matrix
confusionmat(test_labels_01, GpredictionsBeta_test_svm)
