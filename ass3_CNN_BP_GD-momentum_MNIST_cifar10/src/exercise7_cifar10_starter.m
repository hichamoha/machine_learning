%{
Exercise 7:  Plot a few images that are misclassified. Plot the confusion
matrix for the predictions on the test set and compute the precision
and the recall for all digits. Write down the number of parameters
for all layers in the network. Write comments about all plots and
figures.
%}

% argument=2 is how many 10000 images that are loaded. 20000 in this
    % example. Load as much as your RAM can handle.
addpath(genpath('./'));
[x_train, y_train, x_test, y_test, classes] = load_cifar10(2);



%% cifar10_starter
[y_test, pred] = cifar10_starter();

%% Plot the filters the first convolutional layer learns.
convLayer1 = net.layers{2}.params.weights;
figure(1)
for i = 1:size(convLayer1,4)
    subplot(4,4,i)
    imagesc(convLayer1(:,:,3,i))
end


%% Confusion matrix
y_test = double(y_test);
[m, order] = confusionmat(y_test, pred)

%figure(2)
%cm = confusionchart(m, order);
%plotconfusion(y_test, pred)

%% Plot of the misclassified images
%misIndexes = find(:,:,1,pred~=y_test);
%misIndexes = find(pred~=y_test);
figure(33)
x_misIndexes = x_test(:,:,:,pred~=y_test);
y_misIndexes = y_test(pred~=y_test);
pred_misIndexes = pred(pred~=y_test);

i = 4;
imagesc(x_misIndexes(:, :, :, i)/255)
%imshow(x_misIndexes(:, :, :, i)/255)
%colormap(gray);
% title("Ground truth = " + y_misIndexes(i) +...
%     ", prediction = " + pred_misIndexes(i))
title(classes(y_misIndexes(i)) + ", predicted as " + classes(pred_misIndexes(i)))

figure(333)
i = 10;
imagesc(x_misIndexes(:, :, :, i)/255)
%imshow(x_misIndexes(:, :, :, i)/255)
%colormap(gray);
% title("Ground truth = " + y_misIndexes(i) +...
%     ", prediction = " + pred_misIndexes(i))
title(classes(y_misIndexes(i)) + ", predicted as " + classes(pred_misIndexes(i)))

%%
%{
%misIndexes = find(:,:,1, pred~=y_test);
misIndexes = find(pred~=y_test);
figure(3)
for i = 1:2
    %subplot(2,3,i)
    subplot(1,2,i)
    imagesc(x_test(:, :, 1, misIndexes(i))/255)
    %colormap(gray);
    %imagesc(x_misIndexes(:, :, :, i/255))
    str1 = sprintf('Predicted as %d', pred(misIndexes(i)) );
    title( str1)
    
end
%}
