# setup.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. Create table data
# ---------------------------------------------------------

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "sleep_hours": [5, 5, 6, 6, 7, 7, 8, 8, 8, 9],
    "exam_score": [35, 40, 50, 55, 65, 70, 78, 85, 90, 96]
}

df = pd.DataFrame(data)

print("\nFULL DATASET")
print(df)


# ---------------------------------------------------------
# 2. Separate input columns and output column
# ---------------------------------------------------------
# X = input features
# y = target/output value

X = df[["study_hours", "attendance", "sleep_hours"]]
y = df["exam_score"]

print("\nINPUT DATA X")
print(X)

print("\nOUTPUT DATA y")
print(y)


# ---------------------------------------------------------
# 3. Split into training and testing data
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("\nTRAINING INPUT DATA X_train")
print(X_train)

print("\nTRAINING OUTPUT DATA y_train")
print(y_train)

print("\nTESTING INPUT DATA X_test")
print(X_test)

print("\nTESTING OUTPUT DATA y_test")
print(y_test)


# ---------------------------------------------------------
# 4. Create and train Linear Regression model
# ---------------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED")


# ---------------------------------------------------------
# 5. Get final learned m and c values
# ---------------------------------------------------------
# For multiple input columns:
# predicted_score = c + m1(study_hours) + m2(attendance) + m3(sleep_hours)

m = model.coef_
c = model.intercept_

print("\nFINAL FORMULA VALUES")
print("m1 / study_hours coefficient:", m[0])
print("m2 / attendance coefficient:", m[1])
print("m3 / sleep_hours coefficient:", m[2])
print("c / intercept:", c)

print("\nFINAL FORMULA")
print(
    "predicted_score =",
    c,
    "+",
    m[0],
    "*(study_hours) +",
    m[1],
    "*(attendance) +",
    m[2],
    "*(sleep_hours)"
)


# ---------------------------------------------------------
# 6. Make predictions on test data
# ---------------------------------------------------------

y_pred = model.predict(X_test)

print("\nPREDICTED SCORES")
print(y_pred)


# ---------------------------------------------------------
# 7. Compare actual vs predicted values
# ---------------------------------------------------------

result_table = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": y_pred
})

print("\nACTUAL VS PREDICTED TABLE")
print(result_table)


# ---------------------------------------------------------
# 8. Evaluate the model
# ---------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# ---------------------------------------------------------
# 9. Manual prediction using final formula
# ---------------------------------------------------------

study_hours = 7
attendance = 82
sleep_hours = 8

manual_prediction = (
    c
    + m[0] * study_hours
    + m[1] * attendance
    + m[2] * sleep_hours
)

print("\nNEW STUDENT DATA")
print("study_hours:", study_hours)
print("attendance:", attendance)
print("sleep_hours:", sleep_hours)

print("\nMANUAL PREDICTION USING FORMULA")
print("Predicted score:", manual_prediction)


# ---------------------------------------------------------
# 10. Same prediction using sklearn
# ---------------------------------------------------------

new_student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "sleep_hours": [sleep_hours]
})

sklearn_prediction = model.predict(new_student)

print("\nSKLEARN PREDICTION")
print("Predicted score:", sklearn_prediction[0])
