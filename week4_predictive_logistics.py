import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Load logistics dataset
df = pd.read_csv("logistics_dataset.csv")

# Separate features and target
X = df.drop("Delivery_Time_Hours", axis=1)
y = df["Delivery_Time_Hours"]

# Define columns
categorical = [
    "Traffic_Level",
    "Weather_Severity",
    "Vehicle_Type"
]

numerical = [
    "Distance_KM",
    "Warehouse_Delay_Min",
    "Driver_Experience_Years",
    "Package_Weight_KG",
    "Route_Stops"
]

# Preprocessing
preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numerical
    ),
    (
        "cat",
        Pipeline([
            ("imputer",
             SimpleImputer(strategy="most_frequent")),
            ("encoder",
             OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    )
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Linear Regression model
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Random Forest model
random_forest = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42
        )
    )
])

# Train models
linear_model.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Predictions
predictions = random_forest.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Optimization recommendations
print("Optimization Strategies:")
print("1. Dynamic route planning")
print("2. Vehicle and driver allocation")
print("3. Warehouse resource allocation")
print("4. Shipment prioritization")
print("5. Traffic-aware scheduling")
print("6. Transportation cost minimization")
