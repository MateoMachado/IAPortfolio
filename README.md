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

## Predictive Analysis Case Study on Wine Quality

<details>
  <summary>View study case</summary>

### Introduction
In this case study, we focus on a dataset that contains various chemical properties of wines. The dataset provides several attributes of wines, and our goal is to leverage these features to predict a particular attribute of interest.

### Objective
The primary aim is to implement a machine learning model, specifically linear regression, to predict a continuous value in the dataset based on other features. Through this predictive analysis, we intend to understand the relationships between different wine properties.

### Methodology

1. **Data Acquisition and Loading**: The data was loaded using Python's `csv.reader` method.

2. **Data Preprocessing**: This involved multiple steps:
   - Conversion of string columns to float for computational purposes.
   - Calculation of the min and max values for each column to understand data range.
   - Normalization of the dataset to bring values between 0 and 1.
   - Calculation of column means and standard deviations, which can be critical for certain algorithms.
   - Standardization of the dataset using the previously computed means and standard deviations.

3. **Data Splitting**: The dataset was split into a training set and a test set, with the training set containing 60% of the data.

4. **Model Implementation and Evaluation**: We used the Linear Regression model from `sklearn` for this task. After training the model on the training set, its performance was evaluated on the test set using Mean Squared Error (MSE) as the evaluation metric.

### Results

**Basic Dataset Statistics**:
- **Column Min-Max**:
  ``[[1.0, 3.0], [11.03, 14.83], [0.74, 5.8], [1.36, 3.23], [10.6, 30.0], [70.0, 162.0], [0.98, 3.88], [0.34, 5.08], [0.13, 0.66], [0.41, 3.58], [1.28, 13.0], [0.48, 1.71], [1.27, 4.0], [278.0, 1680.0]]
  ``
- **Means**:
  ``[1.9382022471910112, 13.000617977528083, 2.336348314606741, 2.3665168539325854, 19.49494382022472, 99.74157303370787, 2.295112359550562, 2.0292696629213474, 0.36185393258426973, 1.5908988764044953, 5.058089882022473, 0.9574494382022468, 2.6116853932584254, 746.8932584269663]
  ``
- **Standard Deviations**:
  ``[0.7750349899850565, 0.8118265380058577, 1.1171460976144627, 0.2743440090608148, 3.3395637671735052, 14.282483515295668, 0.6258510488339891, 0.9988586850169465, 0.12445334029667939, 0.5723588626747611, 2.318285871822413, 0.22857156582982338, 0.7099904287650505, 314.9074742768489]
  ``


**Linear Regression Performance**:
- **MSE**: 0.02

### Conclusion
The linear regression model achieved an MSE of 0.02, which indicates a fairly low error rate in the predictions. The basic statistics, like column min-max, means, and standard deviations, offer insights into the distribution and variability of the data. This predictive analysis showcases the potential of machine learning in deriving meaningful insights from wine property data and its capability in predicting wine attributes.

<details>
  <summary><strong>Click to view Python code for this analysis</strong></summary>
  
```python
from csv import reader
from math import sqrt
from random import seed, randrange
from sklearn.linear_model import LinearRegression



# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Calculate column means
def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means

# Calculate column standard deviations
def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i]-means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
    return stdevs

# Standardize dataset
def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]

# Split a dataset into a train and test set
def train_test_split(dataset, split=0.60):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

def evaluate_linear_regression(train, test):
    # Extracting features and labels
    train_x = [row[:-1] for row in train]
    train_y = [row[-1] for row in train]

    test_x = [row[:-1] for row in test]
    test_y = [row[-1] for row in test]

    # Creating the regressor
    regressor = LinearRegression()

    # Training the regressor
    regressor.fit(train_x, train_y)

    # Predicting the values of the test set
    predicted = regressor.predict(test_x)

    # Calculating MSE (or another appropriate metric)
    mse = sum([(predicted[i] - test_y[i])**2 for i in range(len(test_y))]) / len(test_y)

    return mse

# Upload the dataset to Google Colab
from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]  # Assuming you've uploaded only one file

dataset = load_csv(filename)

for row in dataset[:10]:
    print(row)

for i in range(len(dataset[0])):  # Assuming all columns should be converted
    str_column_to_float(dataset, i)

minmax = dataset_minmax(dataset)
print("Printing min and max of each column")
print(minmax)  # This will print a list of [min, max] for each column

means = column_means(dataset)
print("Means")
print(means)

stdevs = column_stdevs(dataset, means)
print("Standard deviation of each column")
print(stdevs)

normalize_dataset(dataset, minmax)

dataset_copy = [row.copy() for row in dataset]  # Creating a copy
standardize_dataset(dataset_copy, means, stdevs)

train, test = train_test_split(dataset)
print(f"Training set size: {len(train)}")
print(f"Testing set size: {len(test)}")

train, test = train_test_split(dataset)
mse = evaluate_linear_regression(train, test)
print(f'Linear Regression MSE: {mse:.2f}')
```
</details>

</details>


## Case Study: Predicting Cardiac Events using Logistic Regression
<details>
<summary><strong>View study case</strong></summary>
  
### Introduction

Heart disease remains one of the leading causes of death globally. Timely predictions and interventions can lead to better patient outcomes. Using machine learning, it's possible to utilize historic patient data to predict future cardiac events. For this study, we used a dataset named `cardiac-training.csv` which contains various patient parameters.

### Objective

Our primary goal with this machine learning experiment is to predict whether a patient would experience a second heart attack. This is a binary classification problem where outcomes are classified as 'Si' (Yes) and 'No'.

### Methodology
1. **Data Acquisition and Inspection**: The first step involved reading our dataset and getting an insight into its composition.

   ```python
   import pandas as pd

   input_file = "cardiac-training.csv"
   df = pd.read_csv(input_file, header=0)
   print(df.values)
   ```

2. **Data Preparation**: We segregated our data into features (X) and the target variable (y). The target for our prediction is the '2do_Ataque_Corazon' column.
  ```python
  X = df.loc[:, df.columns != '2do_Ataque_Corazon']
  y = df['2do_Ataque_Corazon'].values
  ```

3. **Training & Testing Split**: To evaluate the model's performance, the dataset was split into training and testing sets, with 70% of the data used for training and 30% reserved for testing.

    ```python
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.30, random_state=0, shuffle=True)
    ```

4. **Model Creation and Training**: A Logistic Regression model was chosen for this binary classification task. The model was then trained on the training data.

  ```python
  from sklearn.linear_model import LogisticRegression
  
  lr = LogisticRegression(max_iter=10000)  # Increased iterations for convergence
  lr = lr.fit(train_X, train_y)
  ```

5. **Model Evaluation**: The trained model was used to predict outcomes on the test data. These predictions were then compared against the true outcomes to evaluate the model's performance.

  ```python
  # Predicting the classes for test data
  y_pred = lr.predict(test_X)
  print("Predicted vs Expected:")
  print(y_pred)
  print(test_y)
  
  # Displaying the classification report
  print("Displaying the classification report")
  print(classification_report(test_y, y_pred, digits=3))
  
  # Displaying the confusion matrix
  print("Displaying the confusion matrix")
  print(confusion_matrix(test_y, y_pred))
  ```

### Results and Conclusion

- Predictions: A snapshot comparison of the predicted vs. expected outcomes was as follows:
  
  ``
  Predicted: ['No', 'Si', 'No', ...]
  ``

  ``
  Expected:  ['No', 'Si', 'No', ...]
  `` 
- Classification Report:

  - Accuracy: The model achieved an accuracy of 90.5% on the test data.
  - Precision for 'No': 94.1%
  - Precision for 'Si': 88.0%
  - Recall for 'No': 84.2%
  - Recall for 'Si': 95.7%

This indicates that our model is fairly reliable in its predictions, with a slight inclination to predict 'Si' more accurately.

- Confusion Matrix:
  
  ``
  [[16  3]
  ``

  ``
  [ 1 22]]
  ``

The model made 3 false positives and 1 false negative predictions.


In conclusion, the Logistic Regression model trained on the cardiac data achieved satisfactory results, making it a potential tool for predicting second cardiac events in patients. However, like any machine learning model, it's essential to continuously train it on new data to adapt to any changing patterns.

</details>


