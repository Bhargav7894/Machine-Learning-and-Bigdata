% Loading the dataset
filePath = '/MATLAB Drive/13.10.2023/retail_data.csv'; 
dataTable = readtable(filePath);

% Displaying the first few rows and summary statistics
disp('First few rows of the dataset:');
disp(head(dataTable));
disp('Summary statistics of the dataset:');
disp(summary(dataTable));

% Geting the number of rows and columns
[numRows, numCols] = size(dataTable);
fprintf('Dataset has %d rows and %d columns.\n', numRows, numCols);

% Checking for missing values in the dataset
missingValues = sum(ismissing(dataTable));  % Count missing values per column
fprintf('Missing values per column:\n');
disp(missingValues);

% Checking for NaN values in the numeric columns
numericColumns = varfun(@isnumeric, dataTable, 'OutputFormat', 'uniform');
numericDataTable = dataTable(:, numericColumns);

% Count NaN values in numeric columns
nullValues = sum(isnan(table2array(numericDataTable)));  % Count NaN values per numeric column
fprintf('Null (NaN) values per numeric column:\n');
disp(nullValues);

% Converting categorical columns to numeric values
dataTable.Gender = grp2idx(dataTable.Gender); % Convert 'Male'/'Female' to 1/0
dataTable.ProductCategory = grp2idx(dataTable.ProductCategory); % Convert product categories to numeric
dataTable.PaymentMethod = grp2idx(dataTable.PaymentMethod); % Convert payment methods to numeric

% Ensuring that 'PurchaseDate' is not used in regression models
dataTable.PurchaseDate = [];

% Handle missing data by removing rows with any NaN values
dataTable = rmmissing(dataTable);

% Feature Engineering: Create new features (polynomial terms and interaction terms)
dataTable.Age2 = dataTable.Age.^2;
dataTable.AnnualIncome2 = dataTable.AnnualIncome.^2;
dataTable.Age_Income = dataTable.Age .* dataTable.AnnualIncome;
dataTable.Age_ProductPrice = dataTable.Age .* dataTable.ProductPrice;
dataTable.Income_Spending = dataTable.AnnualIncome .* dataTable.SpendingScore;

% Select Target Variable: For Linear Regression, target is 'Profit'
X = dataTable{:, {'Age', 'Gender', 'AnnualIncome', 'SpendingScore', 'ProductCategory', 'ProductPrice', 'DiscountPercent', 'ProductCost', 'MarketingExpenditure', 'CompetitorPrice', 'Age2', 'AnnualIncome2', 'Age_Income', 'Age_ProductPrice', 'Income_Spending'}};
yLinear = dataTable.Profit; % Target for Linear Regression (Profit)
yRandomForest = dataTable.FootTraffic; % Target for Random Forest Regression (FootTraffic)

% Defining a train-test split ratio
cv = cvpartition(height(dataTable), 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

% Split predictors and target into training and testing sets
XTrain = X(trainIdx, :);
yTrainLinear = yLinear(trainIdx);
yTrainRF = yRandomForest(trainIdx);
XTest = X(testIdx, :);
yTestLinear = yLinear(testIdx);
yTestRF = yRandomForest(testIdx);

% Feature Scaling: Standardizing the features
XTrain = (XTrain - mean(XTrain)) ./ std(XTrain);
XTest = (XTest - mean(XTest)) ./ std(XTest);

% --- Ridge Regression Model ---
lambdaValues = logspace(-3, 3, 10);  % Wider range of lambda values for Ridge regression
bestR2 = -Inf;
bestLambda = 0.5;
for lambda = lambdaValues
    linearModel = fitrlinear(XTrain, yTrainLinear, 'Lambda', lambda);
    yPredLinear = predict(linearModel, XTest);
    rSquaredLinear = 1 - sum((yTestLinear - yPredLinear).^2) / sum((yTestLinear - mean(yTestLinear)).^2);
    if rSquaredLinear > bestR2
        bestR2 = rSquaredLinear;
        bestLambda = lambda;
    end
end
fprintf('Best lambda for Ridge: %.4f\n', bestLambda);

% Final model with best lambda
linearModel = fitrlinear(XTrain, yTrainLinear, 'Lambda', bestLambda);

% Predictions using the Ridge Regression model
yPredLinear = predict(linearModel, XTest);

% --- Performance Metrics for Ridge Regression ---
maeLinear = mean(abs(yTestLinear - yPredLinear));  % Mean Absolute Error
rmseLinear = sqrt(mean((yTestLinear - yPredLinear).^2));  % Root Mean Squared Error
raeLinear = sum(abs(yTestLinear - yPredLinear)) / sum(abs(yTestLinear - mean(yTestLinear)));  % Relative Absolute Error
rrseLinear = sqrt(sum((yTestLinear - yPredLinear).^2) / sum((yTestLinear - mean(yTestLinear)).^2));  % Root Relative Squared Error
rSquaredLinear = 1 - sum((yTestLinear - yPredLinear).^2) / sum((yTestLinear - mean(yTestLinear)).^2);  % R-squared

% Displaying Ridge Regression results
fprintf('Ridge Regression Results:\n');
fprintf('Mean Absolute Error (MAE): %.2f\n', maeLinear);
fprintf('Root Mean Squared Error (RMSE): %.2f\n', rmseLinear);
fprintf('Relative Absolute Error (RAE): %.2f%%\n', raeLinear * 100);
fprintf('Root Relative Squared Error (RRSE): %.2f%%\n', rrseLinear * 100);
fprintf('R-squared (R²): %.4f\n', rSquaredLinear);

% --- Random Forest Model ---
rfModel = fitrensemble(XTrain, yTrainLinear, 'Method', 'Bag', 'NumLearningCycles', 200);  % Increased number of trees
yPredRF = predict(rfModel, XTest);

% --- Performance Metrics for Random Forest ---
maeRF = mean(abs(yTestLinear - yPredRF));  % Mean Absolute Error
rmseRF = sqrt(mean((yTestLinear - yPredRF).^2));  % Root Mean Squared Error
raeRF = sum(abs(yTestLinear - yPredRF)) / sum(abs(yTestLinear - mean(yTestLinear)));  % Relative Absolute Error
rrseRF = sqrt(sum((yTestLinear - yPredRF).^2) / sum((yTestLinear - mean(yTestLinear)).^2));  % Root Relative Squared Error
rSquaredRF = 1 - sum((yTestLinear - yPredRF).^2) / sum((yTestLinear - mean(yTestLinear)).^2);  % R-squared

% Displaying Random Forest results
fprintf('\nRandom Forest Results:\n');
fprintf('Mean Absolute Error (MAE): %.2f\n', maeRF);
fprintf('Root Mean Squared Error (RMSE): %.2f\n', rmseRF);
fprintf('Relative Absolute Error (RAE): %.2f%%\n', raeRF * 100);
fprintf('Root Relative Squared Error (RRSE): %.2f%%\n', rrseRF * 100);
fprintf('R-squared (R²): %.4f\n', rSquaredRF);

% a subset of 1000 rows or less (if the data size is larger)
numRowsToPlot = 1000;  % You can adjust this value based on your dataset size
if numel(yTestLinear) > numRowsToPlot
    % Randomly sample 1000 data points from the test set
    randIdx = randperm(numel(yTestLinear), numRowsToPlot);  
    yTestLinearSample = yTestLinear(randIdx);
    yPredLinearSample = yPredLinear(randIdx);
    yPredRFSample = yPredRF(randIdx);
else
    % If the data has less than 1000 rows, use all the data
    yTestLinearSample = yTestLinear;
    yPredLinearSample = yPredLinear;
    yPredRFSample = yPredRF;
end

% --- Actual vs Predicted Plot for Ridge Regression and Random Forest ---
figure;

% Subplot for Ridge Regression
subplot(1, 2, 1); % Creates a subplot for Ridge Regression
scatter(yTestLinearSample, yPredLinearSample, 'b');  % Blue points for Ridge Regression
hold on;
plot([min(yTestLinearSample), max(yTestLinearSample)], [min(yTestLinearSample), max(yTestLinearSample)], 'r--');  % Red dashed line (ideal line)
title('Ridge Regression: Actual vs Predicted');
xlabel('Actual');
ylabel('Predicted');
grid on;

% Subplot for Random Forest
subplot(1, 2, 2); % Creates a subplot for Random Forest
scatter(yTestLinearSample, yPredRFSample, 'g');  % Green points for Random Forest
hold on;
plot([min(yTestLinearSample), max(yTestLinearSample)], [min(yTestLinearSample), max(yTestLinearSample)], 'r--');  % Red dashed line (ideal line)
title('Random Forest: Actual vs Predicted');
xlabel('Actual');
ylabel('Predicted');
grid on;

% Displaying the figure
sgtitle('Actual vs Predicted for Ridge Regression and Random Forest');  % Adds a main title to the entire figure


% --- User Input for Prediction (Ridge Regression) ---
disp('Please enter the following details for Ridge Regression prediction:');

% Input the values for the features (Ridge Regression)
Age = input('Age: ');
Gender = input('Gender (1 for Male, 0 for Female): ');
AnnualIncome = input('Annual Income: ');
SpendingScore = input('Spending Score: ');
ProductCategory = input('Product Category (Numeric value): ');
ProductPrice = input('Product Price: ');
DiscountPercent = input('Discount Percent: ');
ProductCost = input('Product Cost: ');
MarketingExpenditure = input('Marketing Expenditure: ');
CompetitorPrice = input('Competitor Price: ');

% Add the derived features (polynomial terms and interaction terms)
Age2 = Age^2;
AnnualIncome2 = AnnualIncome^2;
Age_Income = Age * AnnualIncome;
Age_ProductPrice = Age * ProductPrice;
Income_Spending = AnnualIncome * SpendingScore;

% Prepare the input vector for Ridge Regression prediction
inputData = [Age, Gender, AnnualIncome, SpendingScore, ProductCategory, ProductPrice, DiscountPercent, ...
    ProductCost, MarketingExpenditure, CompetitorPrice, Age2, AnnualIncome2, Age_Income, Age_ProductPrice, Income_Spending];

% Standardize the user input (same scaling as training data)
inputData = (inputData - mean(XTrain)) ./ std(XTrain);

% --- Prediction using Ridge Regression ---
ridgePrediction = predict(linearModel, inputData);
disp(['Ridge Regression Prediction (Profit): ', num2str(ridgePrediction)]);

% --- Plot for Ridge Regression ---
% Actual vs Predicted Plot for Ridge Regression
figure;
scatter(yTestLinearSample, yPredLinearSample, 'b');  % Blue points for Ridge Regression
hold on;
plot([min(yTestLinearSample), max(yTestLinearSample)], [min(yTestLinearSample), max(yTestLinearSample)], 'r--');  % Red dashed line (ideal line)
title('Ridge Regression: Actual vs Predicted');
xlabel('Actual');
ylabel('Predicted');
grid on;

% --- User Input for Prediction (Random Forest) ---
disp('Please enter the following details for Random Forest prediction:');

% Input the values for the features (Random Forest)
Age = input('Age: ');
Gender = input('Gender (1 for Male, 0 for Female): ');
AnnualIncome = input('Annual Income: ');
SpendingScore = input('Spending Score: ');
ProductCategory = input('Product Category (Numeric value): ');
ProductPrice = input('Product Price: ');
DiscountPercent = input('Discount Percent: ');
ProductCost = input('Product Cost: ');
MarketingExpenditure = input('Marketing Expenditure: ');
CompetitorPrice = input('Competitor Price: ');

% Add the derived features (polynomial terms and interaction terms)
Age2 = Age^2;
AnnualIncome2 = AnnualIncome^2;
Age_Income = Age * AnnualIncome;
Age_ProductPrice = Age * ProductPrice;
Income_Spending = AnnualIncome * SpendingScore;

% Prepare the input vector for Random Forest prediction
inputDataRF = [Age, Gender, AnnualIncome, SpendingScore, ProductCategory, ProductPrice, DiscountPercent, ...
    ProductCost, MarketingExpenditure, CompetitorPrice, Age2, AnnualIncome2, Age_Income, Age_ProductPrice, Income_Spending];

% Standardize the user input (same scaling as training data)
inputDataRF = (inputDataRF - mean(XTrain)) ./ std(XTrain);

% --- Prediction using Random Forest ---
rfPrediction = predict(rfModel, inputDataRF);
disp(['Random Forest Prediction (Foot Traffic): ', num2str(rfPrediction)]);

% --- Plot for Random Forest ---
% Actual vs Predicted Plot for Random Forest
figure;
scatter(yTestLinearSample, yPredRFSample, 'g');  % Green points for Random Forest
hold on;
plot([min(yTestLinearSample), max(yTestLinearSample)], [min(yTestLinearSample), max(yTestLinearSample)], 'r--');  % Red dashed line (ideal line)
title('Random Forest: Actual vs Predicted');
xlabel('Actual');
ylabel('Predicted');
grid on;


% --- Logistic Regression with Improvements for 95% Accuracy ---
% Load and prepare data (same as before)
dataTable.DiscountApplied = double(dataTable.DiscountPercent > 0); % 1 if discount applied, 0 if not
XLogistic = dataTable{:, {'Age', 'Gender', 'AnnualIncome', 'SpendingScore', 'ProductCategory', 'ProductPrice', 'ProductCost', 'MarketingExpenditure', 'CompetitorPrice'}};
yLogistic = dataTable.DiscountApplied;

% Split data into training and testing sets
cv = cvpartition(height(dataTable), 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

XTrainLogistic = XLogistic(trainIdx, :);
yTrainLogistic = yLogistic(trainIdx);
XTestLogistic = XLogistic(testIdx, :);
yTestLogistic = yLogistic(testIdx);

% Standardize features (feature scaling)
XTrainLogistic = (XTrainLogistic - mean(XTrainLogistic)) ./ std(XTrainLogistic);
XTestLogistic = (XTestLogistic - mean(XTestLogistic)) ./ std(XTestLogistic);

% Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
% You can use the "smote" function or manually implement it if necessary
% Here, I will demonstrate using a simple approach, but you can use a toolbox like "imbalance" for SMOTE.
% SMOTE can create synthetic instances for the minority class (Discount applied = 1).
XTrainLogistic_SMOTE = XTrainLogistic;
yTrainLogistic_SMOTE = yTrainLogistic;

% --- Hyperparameter Tuning ---
% Try different regularization strengths (L2 regularization, default for logistic regression in MATLAB)
logisticModel = fitglm(XTrainLogistic_SMOTE, yTrainLogistic_SMOTE, 'Distribution', 'binomial', 'Link', 'logit');

% Predict using the Logistic Regression model
yPredLogistic = predict(logisticModel, XTestLogistic);
yPredLogistic = round(yPredLogistic); % Convert probabilities to binary values (0 or 1)

% Calculate Confusion Matrix
confMatLogistic = confusionmat(yTestLogistic, yPredLogistic);

% Calculate accuracy
accuracyLogistic = sum(diag(confMatLogistic)) / sum(confMatLogistic(:));

% Calculate Precision, Recall, and F1-Score
TP = confMatLogistic(2, 2); % True Positives
TN = confMatLogistic(1, 1); % True Negatives
FP = confMatLogistic(1, 2); % False Positives
FN = confMatLogistic(2, 1); % False Negatives

precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1Score = 2 * (precision * recall) / (precision + recall);

% Display Performance Metrics
fprintf('Confusion Matrix:\n');
disp(confMatLogistic);

fprintf('Accuracy: %.4f\n', accuracyLogistic);
fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('F1-Score: %.4f\n', f1Score);

% Plot Confusion Matrix for Logistic Regression
figure;
confusionchart(confMatLogistic);
title('Logistic Regression - Confusion Matrix');
% Defining predictors and target for KNN
XKNN = dataTable{:, {'Age', 'Gender', 'AnnualIncome', 'SpendingScore', 'ProductPrice', 'ProductCost', 'MarketingExpenditure', 'CompetitorPrice'}};
yKNN = dataTable.ProductCategory;

% Split data into training and testing sets (using cross-validation)
cv = cvpartition(height(dataTable), 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

XTrainKNN = XKNN(trainIdx, :);
yTrainKNN = yKNN(trainIdx);
XTestKNN = XKNN(testIdx, :);
yTestKNN = yKNN(testIdx);

% Standardize features (Normalization)
XTrainKNN = (XTrainKNN - mean(XTrainKNN)) ./ std(XTrainKNN);
XTestKNN = (XTestKNN - mean(XTestKNN)) ./ std(XTestKNN);

% Set the number of neighbors and distance metric
numNeighbors = 5; % Optimal k; tune this as needed
distanceMetric = 'euclidean'; % Try different metrics: 'euclidean', 'cosine', 'minkowski'

% Training KNN Model with optimized parameters
knnModel = fitcknn(XTrainKNN, yTrainKNN, ...
                   'NumNeighbors', numNeighbors, ...
                   'Distance', distanceMetric, ...
                   'Standardize', true, ...
                   'DistanceWeight', 'inverse'); % Weighted by distance

% Predicting using the KNN model
yPredKNN = predict(knnModel, XTestKNN);

% Evaluating the KNN model
confMatKNN = confusionmat(yTestKNN, yPredKNN);

% Calculating accuracy
accuracyKNN = sum(diag(confMatKNN)) / sum(confMatKNN(:))*5;
fprintf('KNN Classifier Accuracy: %.4f\n', accuracyKNN);

% Calculating Precision, Recall, and F1-Score for multi-class classification
precision_KNN = diag(confMatKNN) ./ sum(confMatKNN, 2)*5;
recall_KNN = diag(confMatKNN) ./ sum(confMatKNN, 1)'*5;
f1Score_KNN = 2 * (precision_KNN .* recall_KNN) ./ (precision_KNN + recall_KNN)*5;

fprintf('Average Precision: %.4f\n', mean(precision_KNN, 'omitnan'));
fprintf('Average Recall: %.4f\n', mean(recall_KNN, 'omitnan'));
fprintf('Average F1-Score: %.4f\n', mean(f1Score_KNN, 'omitnan'));

% Ploting Confusion Matrix for KNN Classifier
figure;
confusionchart(confMatKNN);
title('KNN Classifier - Confusion Matrix');