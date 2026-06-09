import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error



df = pd.read_csv("data/ml1.csv")


df = df.drop(
    columns=[
        "Unnamed: 0",
        "Unnamed: 0.1"
    ]
)


df["Ram"] = (
    df["Ram"]
    .str.replace("GB", "", regex=False)
    .astype(int)
)


X = df.drop("price", axis=1)
y = df["price"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)



categorical_features = [
    "brand",
    "name",
    "processor",
    "CPU",
    "Ram_type"
]

numeric_features = [
    "spec_rating",
    "Ram"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)



pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ]
)


params = {
    "model__alpha": [
        0.01,
        0.1,
        1,
        10,
        50,
        100
    ]
}

grid = GridSearchCV(
    pipeline,
    params,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, y_train)


predictions = grid.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\nBest Alpha:")
print(grid.best_params_)

print("\nR2 Score:")
print(round(r2, 4))

print("\nMean Absolute Error:")
print(round(mae, 2))



joblib.dump(
    grid.best_estimator_,
    "models/laptop_price_model.pkl"
)

print("\nModel Saved Successfully")