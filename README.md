# Machine Learning Portfolio
## Overview
This portfolio is a curated collection of projects and exercises from the "Introduction to Automatic Learning Methods" (Introducción a los Métodos de Aprendizaje Automático) course at Universidad Católica del Uruguay.

## About Me
Hello! I'm Mateo Machado, a Computer Science Engineering student at Universidad Católica del Uruguay. While my journey in machine learning is academic at its core, this portfolio represents the skills and understanding I've developed throughout the course. Here, you'll find practical applications and exercises carried out using tools such as RapidMiner and Python.

## Portfolio Goals
This space is designed to showcase my grasp on machine learning concepts, algorithms, and tools as covered in my course. It provides a glimpse into the academic requirements of the subject and my efforts in navigating through them.

## Mapping ML Capabilities

Throughout my academic experience in the "Introduction to Automatic Learning Methods" course, I've cultivated an array of machine learning skills and competencies. Below is an in-depth exploration of these capabilities:

### Data Preparation

- **Preprocessing**: Utilizing tools and libraries to cleanse, normalize, and transform raw data, preparing it for machine learning workflows.
  
- **Handling Missing Data**: Implementing various imputation methods, like mean imputation or model-driven imputation, to address gaps in datasets.
  
- **Outlier Detection**: Employing statistical methods and visualization tools to identify and manage anomalies in datasets, ensuring robust model performance.
  
- **Distribution Analysis**: Analyzing the statistical distribution of datasets, including skewness and kurtosis, and making informed decisions about normalization or transformation techniques, such as logarithmic or Box-Cox transformations.
  
- **Feature Selection and Engineering**: Using correlation matrices to identify and remove correlated features, enhancing model interpretability and performance. Additionally, deriving new features based on existing data to offer improved predictive insights.
  
- **Removing Correlated Columns**: Actively monitoring and removing features with high multicollinearity to prevent overfitting and improve model generalization.

### Criteria for Choosing Algorithms & Techniques

- **Naive Bayes**:
  - **Use Cases**: Best suited for classification tasks, particularly when features are categorical.
  - **Criteria**: Requires the data to be normalized. Assumes that features are independent, a condition called conditional independence.
  
- **Logistic Regression**:
  - **Use Cases**: Implemented for problems where the outcome is binary, such as spam detection or customer churn prediction.
  - **Criteria**: Assumes a linear relationship between the predictors and the log odds of the outcome. Data should not have high multicollinearity.
  
- **Linear Regression**:
  - **Use Cases**: Applied to predict a continuous outcome variable based on one or more predictor variables.
  - **Criteria**: Assumes linearity, homoscedasticity, and independence of errors. Sensitive to outliers and multicollinearity.
  
- **LDA (Linear Discriminant Analysis)**:
  - **Use Cases**: Deployed both for modeling differences in groups and for dimensionality reduction.
  - **Criteria**: Assumes that the independent variables are normally distributed and have the same variance in each group.
  
- **k-Nearest Neighbors**:
  - **Use Cases**: Employed for both classification and regression tasks. Especially useful when decision boundaries are non-linear.
  - **Criteria**: Requires normalized data as it relies on distance calculations. Performance can degrade with high-dimensional data unless dimensionality reduction techniques are applied.

### Tools & Practical Implementations

- **RapidMiner**: An integral data science platform for diverse tasks, encompassing data preparation to model deployment.

- **Python**: A flexible scripting language, enhanced with libraries like scikit-learn, for a wide spectrum of machine learning applications.

# Cross-sectional Case Study: Predictive Analysis on Wine Quality

## Introduction:

The wine industry has always been associated with the quality of its products. Factors such as the alcohol content, acidity, and residual sugar can all contribute to the taste, longevity, and overall appreciation of wine. The given dataset provides numeric features representing these factors among others, making it a suitable candidate for predictive modeling to understand the determinants of wine quality.

## Objective:

The primary aim is to leverage machine learning, specifically Linear Regression in this case, to understand the relationship between various attributes of wine and its quality. This predictive model can potentially serve as a tool for vintners to improve their wine production process by focusing on attributes that significantly impact quality.

## Methodology:

1. **Data Acquisition:** 
   The data was uploaded to the working environment using the Google Colab platform's file upload feature.

\```python
from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]
\```

2. **Data Preprocessing:** 
   - Loaded the dataset from a CSV file.
   - Converted the string values in the dataset to float for computational purposes.
   - Determined and printed the minimum and maximum values for each column.
   - Calculated the mean and standard deviation for each column.
   - Normalized the dataset to scale features between 0 and 1.
   - Standardized the dataset using the calculated mean and standard deviation.

\```python
dataset = load_csv(filename)
for i in range(len(dataset[0])): 
    str_column_to_float(dataset, i)

minmax = dataset_minmax(dataset)
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)

normalize_dataset(dataset, minmax)
dataset_copy = [row.copy() for row in dataset]
standardize_dataset(dataset_copy, means, stdevs)
\```

3. **Data Splitting:** 
   The dataset was split into training and testing sets, using a 60:40 ratio.

\```python
train, test = train_test_split(dataset)
\```

4. **Model Training and Evaluation:** 
   - Utilized the `LinearRegression` model from Scikit-Learn.
   - Extracted features and target labels from the training set.
   - Trained the linear regression model on the training set.
   - Predicted wine quality on the testing set.
   - Evaluated the model's performance using the Mean Squared Error (MSE) metric.

\```python
mse = evaluate_linear_regression(train, test)
\```

## Results and Conclusion:

- **Data Overview:**
  Here's a snapshot of the first few rows of the dataset:
  \```
  ['1', '14.23', '1.71', ... '3.92', '1065']
  ['1', '13.2', '1.78', ... '3.4', '1050']
  ...
  \```

- **Statistical Overview:**
  - Minimum and Maximum of each column:
  \```
  [[1.0, 3.0], [11.03, 14.83], ... [278.0, 1680.0]]
  \```
  - Mean of each column:
  \```
  [1.938, 13.001, ... 2.611, 746.893]
  \```
  - Standard deviation of each column:
  \```
  [0.775, 0.812, ... 0.710, 314.907]
  \```

- **Model Metrics:** 
  The trained Linear Regression model achieved an MSE of 0.02. This low value indicates that the model has done a relatively good job of predicting wine quality, with minimal errors between the predicted and actual values.

- **Insights Gained:**
  The model's low MSE suggests that the features in the dataset have a significant impact on determining wine quality. This insight can be valuable for wine producers, enabling them to prioritize factors that most influence wine quality.

- **Concluding Remarks:** 
  Machine Learning, particularly regression analysis, provides an effective approach to understanding the determinants of wine quality. Future studies could delve deeper by employing more complex models, feature engineering, or even looking into different evaluation metrics to gain a broader understanding of the wine production process and quality determinants.


