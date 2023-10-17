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

    # Calculating MSE 
    mse = sum([(predicted[i] - test_y[i])**2 for i in range(len(test_y))]) / len(test_y)

    return mse

# Upload the dataset to Google Colab
from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]

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