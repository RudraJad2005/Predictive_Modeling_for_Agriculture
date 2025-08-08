
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Loading the dataset
crops = pd.read_csv("Data_Set.csv")

# Data Preprocessing
# print(crops.head())
# print(crops.info())
# print(crops.describe())

# Converting categorical data to numerical data
crops = pd.get_dummies(crops, drop_first=True)

features = ["N", "P", "K", "ph"]

best_score = 0
best_feature = None

for feature in features:
    X = crops.drop(feature, axis=1)
    y = crops[feature]

    # Binarizing the target variable
    y = (y > y.median()).astype(int)

    # Training the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiating the model with scaling and more iterations to help convergence
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs")
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = metrics.balanced_accuracy_score(y_test, y_pred)
    
    if score > best_score:
        best_score = score
        best_feature = feature
    
    print(f"{feature} {score:.10f}")

print(f"Best predictive feature: {best_feature} with a score of {best_score:.10f}")

best_predictive_feature = {
    best_feature: best_score
}