# Import the breast cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer

# Import logistic regression model
from sklearn.linear_model import LogisticRegression

# Import StandardScaler to scale the feature values
from sklearn.preprocessing import StandardScaler

# Import metrics to check model performance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import train_test_split to split data into training and testing parts
from sklearn.model_selection import train_test_split

# Import pandas if we want to use dataframes later
import pandas as pd


# Create the logistic regression model
model = LogisticRegression(max_iter=1000)

# Load the breast cancer dataset
data = load_breast_cancer()


# Store input features in X
# X contains all tumor measurement values
X = data.data
print(X)

# Store target labels in y
# y contains the actual class labels
# 0 = malignant, 1 = benign
y = data.target
print(y)


# Split the dataset into training and testing data
# 80% of the data is used for training
# 20% of the data is used for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train the logistic regression model using unscaled training data
model.fit(X_train, y_train)

# Predict the test data using the unscaled model
y_pred = model.predict(X_test)
print(y_pred)


# Create a StandardScaler object
# It will scale the feature values
scaler = StandardScaler()

# Fit the scaler on training data and scale the training data
# fit_transform finds the mean and standard deviation from X_train
X_train_scaled = scaler.fit_transform(X_train)

# Scale the test data using the same mean and standard deviation from training data
X_test_scaled = scaler.transform(X_test)


# Train the logistic regression model again using scaled training data
model.fit(X_train_scaled, y_train)

# Predict the test data using the scaled model
y_pred_scaled = model.predict(X_test_scaled)
print(y_pred_scaled)


# Calculate the accuracy of the scaled model
# Accuracy means correct predictions divided by total predictions
accuracy = accuracy_score(y_test, y_pred_scaled)
print("Accuracy:", accuracy)


# Create the confusion matrix
# It shows correct and incorrect predictions for each class
conf_matrix = confusion_matrix(y_test, y_pred_scaled)
print("Confusion Matrix:", conf_matrix)


# Create the classification report
# It shows precision, recall, f1-score, and support
class_report = classification_report(y_test, y_pred_scaled)
print("Classification Report:", class_report)


# Take one sample from the test data
sample = X_test[0]

# Scale the sample before prediction
# The model was trained on scaled data, so new input must also be scaled
sample_scaled = scaler.transform([sample])

# Predict the class of the sample
prediction = model.predict(sample_scaled)

# Get the probability for each class
probability = model.predict_proba(sample_scaled)


# Print the actual class of the sample
print("\nOne sample actual class:", data.target_names[y_test[0]])

# Print the predicted class of the sample
print("One sample predicted class:", data.target_names[prediction[0]])

# Print the prediction probabilities
print("Prediction probabilities:", probability)


# Print the intercept learned by the model
print("\nIntercept:")
print(model.intercept_)


# Print how many slopes or weights the model learned
# There is one weight for each feature
print("\nNumber of slopes/weights learned:")
print(len(model.coef_[0]))


# Print the first 5 learned weights
print("\nFirst 5 slopes/weights:")
print(model.coef_[0][:5])