%{
Exercise 6:  Plot a few images that are misclassified. Plot the confusion
matrix for the predictions on the test set and compute the precision
and the recall for all digits. Write down the number of parameters
for all layers in the network. Write comments about all plots and
figures.
%}
x_test = loadMNISTImages('data/mnist/t10k-images.idx3-ubyte');
x_test = reshape(x_test, [28, 28, 1, 10000]);
y_test = loadMNISTLabels('data/mnist/t10k-labels.idx1-ubyte');


%% mnist_starter
[y_test, pred] = mnist_starter();

%% Plot the filters the first convolutional layer learns.
convLayer1 = net.layers{2}.params.weights;
figure(1)
for i = 1:size(convLayer1,4)
    subplot(4,4,i)
    imagesc(convLayer1(:,:,1,i))
end

%% Confusion matrix
%[m, order] = confusionmat(y_test, pred)

figure(2)
%cm = confusionchart(m, order);
plotconfusion(y_test, pred)

%% Plot of the misclassified images
%missIndexes = find(:,:,1,pred~=y_test);
missIndexes = find(pred~=y_test);

% x_missIndexes = x_test(:,:,1,pred~=y_test);
% y_missIndexes = y_test(pred~=y_test);
% pred_missIndexes = pred(pred~=y_test);

figure(3)
for i = 1:4
    %subplot(2,3,i)
    subplot(2,2,i)
    imagesc(x_test(:, :, 1, missIndexes(i)))
    %imagesc(x_missIndexes(:, :, 1, i))
    str1 = sprintf('Predicted as %d', pred(missIndexes(i)) );
    title(str1)
end
