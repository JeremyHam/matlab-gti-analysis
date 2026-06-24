%/***********************************************************************
% ECE 552 - Visual Perception for Autonomy
% Final Project - Vehicle Classification Using HOG and LBP Feature Fusion
% 
% This project looks at classifying vehicle and non-vehicle images using
% HOG and LBP feature vectors individually, both features using feature
% fusion, and if PCA reduces classification accuracy.
%
% This project uses the GTI Cars Dataset, which can be downloaded here:
%       https://www.kaggle.com/datasets/iamprateek/vehicle-images-gti
%************************************************************************\
clear; close all; clc;
%% Import Dataset

% Set the data paths for the GTI dataset
vehicles = "VehicleImage\vehicles";
nonVehicles = "VehicleImage\non-vehicles";

% Load vehicle and non-vehicle images from all the subfolders
vehicleFiles = dir(fullfile(vehicles, '**', '*.png'));
nonVehicleFiles = dir(fullfile(nonVehicles, '**', '*.png'));

% Create the full image path for the images
vehicleImagePaths = fullfile({vehicleFiles.folder}, {vehicleFiles.name});
nonVehicleImagePaths = fullfile({nonVehicleFiles.folder}, {nonVehicleFiles.name});

% Combine the images and create labels for the SVM
images = [vehicleImagePaths, nonVehicleImagePaths]';
labels = [ones(numel(vehicleImagePaths), 1); zeros(numel(nonVehicleImagePaths), 1)];

% Create an 80/20 train/test split
rng(1); % For reproducibility
cv = cvpartition(labels, "HoldOut", 0.2);
trainIdx = training(cv);
testIdx = test(cv);

% Split the images and labels
xTrain = images(trainIdx);
yTrain = labels(trainIdx);
xTest = images(testIdx);
yTest = labels(testIdx);

%% HOG-Only Classification

% Define parameters for HOG feature extraction
cellSize = [8,8];
blocks = [16,16];
nBins = 9;
imRows = 64;
imCols = 192;
blockSize = ceil(blocks./cellSize);
overlapSize = ceil(blockSize/2);

% Determine the feature vector length
img = imread(string(xTrain{1}));
img = rgb2gray(imresize(img, [imRows, imCols]));
feature_dims = length(extractHOGFeatures(img, 'CellSize', cellSize, 'BlockSize', blockSize, ...
        'BlockOverlap', overlapSize, 'NumBins', nBins));

trainHOG = zeros(length(xTrain), feature_dims);
testHOG = zeros(length(xTest), feature_dims);

% Get the HOG feature vectors for all training images
for i = 1:length(xTrain)
    img = imread(string(xTrain{i}));
    img = rgb2gray(imresize(img, [imRows, imCols]));
    trainHOG(i, :) = extractHOGFeatures(img, 'CellSize', cellSize, 'BlockSize', blockSize, ...
        'BlockOverlap', overlapSize, 'NumBins', nBins);
end

% Get the HOG feature vectors for all testing images
for i = 1:length(xTest)
    img = imread(string(xTest{i}));
    img = rgb2gray(imresize(img, [imRows, imCols]));
    testHOG(i, :) = extractHOGFeatures(img, 'CellSize', cellSize, 'BlockSize', blockSize, ...
        'BlockOverlap', overlapSize, 'NumBins', nBins);
end

% Train an SVM model using the training HOG set
svmModel = fitcsvm(trainHOG, yTrain, 'KernelFunction', 'linear', 'Standardize', true);

% Test the SVM model using the testing HOG set
predictions = predict(svmModel, testHOG);

% Calculate the accuracy of the SVM model
accuracy = sum(predictions == yTest) / length(yTest);
fprintf('Accuracy of HOG-only classification: %.2f%%\n', accuracy * 100);

% Create a confusion matrix for the HOG-only results
confusionMat = confusionmat(yTest, predictions);
figure;
heatmap({'0', '1'}, {'0', '1'}, confusionMat, 'XLabel', 'Predicted', 'YLabel', 'True', ...
         'Title', 'Confusion Matrix for HOG-only Classification');

%% LBP-Only Classification

% Define the LBP parameters
radius = 1;
numNeighbors = 8;

% Determine the length of the LBP feature vectors
img = imread(string(xTrain{1}));
img = rgb2gray(imresize(img, [imRows, imCols]));
featureDim = length(extractLBPFeatures(img, 'Radius', radius, 'NumNeighbors', numNeighbors));
trainLBP = zeros(length(xTrain), featureDim);
testLBP = zeros(length(xTest), featureDim);

% Get the LBP vectors for the training images
for i = 1:length(xTrain)
    img = imread(string(xTrain{i}));
    img = rgb2gray(imresize(img, [imRows, imCols]));
    lbpVector = extractLBPFeatures(img, 'Radius', radius, 'NumNeighbors', numNeighbors);
    trainLBP(i, :) = lbpVector;
end

% Get the LBP vectors for the testing images
for i = 1:length(xTest)
    img = imread(string(xTest{i}));
    img = rgb2gray(imresize(img, [imRows, imCols]));
    lbpVector = extractLBPFeatures(img, 'Radius', radius, 'NumNeighbors', numNeighbors);
    testLBP(i, :) = lbpVector;
end

% Train an SVM model using the training LBP set
svmModelLBP = fitcsvm(trainLBP, yTrain, 'KernelFunction', 'linear', 'Standardize', true);

% Test the SVM model using the testing LBP set
lbpPredictions = predict(svmModelLBP, testLBP);

% Calculate the accuracy of the LBP model
lbpAccuracy = sum(lbpPredictions == yTest) / length(yTest);
fprintf('Accuracy of LBP-only classification: %.2f%%\n', lbpAccuracy * 100);

% Create a confusion matrix for the LBP-only results
confusionMat = confusionmat(yTest, lbpPredictions);
figure;
heatmap({'0', '1'}, {'0', '1'}, confusionMat, 'XLabel', 'Predicted', 'YLabel', 'True', ...
         'Title', 'Confusion Matrix for LBP-only Classification');

%% HOG + LBP Feature Fusion (No PCA)

% Concatenate HOG and LBP features for training and testing
trainFeatures = [trainHOG, trainLBP];
testFeatures = [testHOG, testLBP];

% Train an SVM model using the fused feature set
svmModelFusion = fitcsvm(trainFeatures, yTrain, 'KernelFunction', 'linear', 'Standardize', true);

% Test the SVM model using the fused feature set
fusionPredictions = predict(svmModelFusion, testFeatures);

% Calculate the accuracy of the fused model
fusionAccuracy = sum(fusionPredictions == yTest) / length(yTest);
fprintf('Accuracy of HOG + LBP fusion classification: %.2f%%\n', fusionAccuracy * 100);

% Create a confusion matrix for the fused model results
fusionConfusionMat = confusionmat(yTest, fusionPredictions);
figure;
heatmap({'0', '1'}, {'0', '1'}, fusionConfusionMat, 'XLabel', 'Predicted', 'YLabel', 'True', ...
         'Title', 'Confusion Matrix for HOG + LBP Fusion Classification');

%% HOG + LBP Feature Fusion (With PCA)

% Apply PCA to reduce dimensionality of the fused feature set
[coeff, trainFeaturesPCA, ~, ~, explained, A] = pca(trainFeatures);

% Find a value L
L = 100;

% Apply the value L to the training features
trainFeaturesPCA = trainFeaturesPCA(:, 1:L);

% Calculate the PCA feature vectors for the testing set
testFeaturesPCA = (testFeatures - A) * coeff(:, 1:L);

% Train an SVM model using the fused feature set
svmModelFusionPCA = fitcsvm(trainFeaturesPCA, yTrain, 'KernelFunction', 'linear', 'Standardize', true);

% Test the SVM model using the fused feature set
fusionPCAPredictions = predict(svmModelFusionPCA, testFeaturesPCA);

% Calculate the accuracy of the fused model
fusionPCAAccuracy = sum(fusionPCAPredictions == yTest) / length(yTest);
fprintf('Accuracy of HOG + LBP fusion classification: %.2f%%\n', fusionPCAAccuracy * 100);

% Create a confusion matrix for the fused model results
fusionPCAConfusionMat = confusionmat(yTest, fusionPCAPredictions);
figure;
heatmap({'0', '1'}, {'0', '1'}, fusionPCAConfusionMat, 'XLabel', 'Predicted', 'YLabel', 'True', ...
         'Title', 'Confusion Matrix for HOG + LBP Fusion With PCA Classification');
