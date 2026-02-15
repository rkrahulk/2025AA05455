import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

LABELS = {0: "Malignant", 1: "Benign"}

st.title("ML Classification Models - Cancer Dataset")

uploaded = st.file_uploader("Upload Test CSV", type=["csv"])

model_name = st.selectbox("Select Model", [
    "Logistic_Regression",
    "Decision_Tree",
    "KNN",
    "Naive_Bayes",
    "Random_Forest",
    "XGBoost"
])

if uploaded:
    df = pd.read_csv(uploaded)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = pickle.load(open("model/"+model_name + ".pkl","rb"))
    preds = model.predict(X)

    st.subheader("Metrics")
    results = pd.read_csv("results.csv")
    st.table(results[results['Model'].str.replace(' ','_')==model_name])

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Malignant","Benign"],
                yticklabels=["Malignant","Benign"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
