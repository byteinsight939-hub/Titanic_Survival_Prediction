import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Titanic Dashboard", layout="wide")

st.title("🚢 Titanic Survival Prediction Dashboard")

# -----------------------
# Data Preprocessing
# -----------------------
def clean_data(df):
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    if "Fare" in df.columns:
        df["Fare"].fillna(df["Fare"].median(), inplace=True)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    df["Title"] = df["Title"].replace(["Mlle", "Ms", "Lady", "Countess", "Dona"], "Miss")
    df["Title"] = df["Title"].replace(["Mme"], "Mrs")
    df["Title"] = df["Title"].replace(["Capt","Col","Major","Dr","Rev","Sir","Don","Jonkheer"], "Rare")

    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])
    df["Title"] = le.fit_transform(df["Title"])

    df = df.drop(["Name","Ticket","Cabin","PassengerId"], axis=1, errors="ignore")
    return df

# -----------------------
# Upload Data
# -----------------------
uploaded_file = st.file_uploader("Upload Titanic CSV (train.csv)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔎 Raw Data Preview")
    st.dataframe(data.head())

    # Clean data
    df = clean_data(data)

    # -----------------------
    # EDA
    # -----------------------
    st.subheader("📊 Exploratory Data Analysis")
    fig, ax = plt.subplots()
    sns.countplot(x="Survived", hue="Sex", data=data, ax=ax)
    st.pyplot(fig)

    # -----------------------
    # Model Training
    # -----------------------
    if "Survived" in df.columns:
        X = df.drop("Survived", axis=1)
        y = df["Survived"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(kernel="linear"),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        results = {}
        trained_models = {}
        for name, model in models.items():
            if name in ["KNN", "SVM"]:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            results[name] = acc
            trained_models[name] = model

        st.subheader("⚖️ Model Comparison")
        st.bar_chart(pd.Series(results))

        best_model_name = max(results, key=results.get)
        st.success(f"✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.2f}")
        best_model = trained_models[best_model_name]

        # -----------------------
        # Interactive Prediction
        # -----------------------
        st.subheader("🧑 Predict Survival for a Passenger")

        pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 30)
        sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
        parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
        fare = st.number_input("Ticket Fare", 0, 600, 50)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

        input_df = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [0 if sex == "male" else 1],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [0 if embarked=="C" else 1 if embarked=="Q" else 2],
            "FamilySize": [sibsp + parch + 1],
            "IsAlone": [1 if (sibsp + parch) == 0 else 0],
            "Title": [1]  # dummy, because model expects column
        })

        # Scale if needed
        if best_model_name in ["KNN", "SVM"]:
            input_scaled = scaler.transform(input_df)
            prediction = best_model.predict(input_scaled)
        else:
            prediction = best_model.predict(input_df)

        st.write("🔮 Prediction:", "Survived 🟢" if prediction[0] == 1 else "Did not survive 🔴")
