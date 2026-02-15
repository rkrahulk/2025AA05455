import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer


def load_and_preprocess(path=None):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # save test set for streamlit upload
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df["target"] = y_test.values
    test_df.to_csv("test_data.csv", index=False)

    return X_train, X_test, y_train, y_test