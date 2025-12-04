import os

import numpy as np
import pandas as pd


def get_data(dataset_name):
    if dataset_name == "admission":
        df = load_admission()
    elif dataset_name == "churn":
        df = load_churn()
    elif dataset_name == "fraud":
        df = load_fraud()
    elif dataset_name == "COMPAS":
        df = load_COMPAS()
    elif dataset_name == "gtd":
        df = load_GTD()
    else:
        raise ValueError("Unknown dataset")
    return df


def load_admission():
    raw_path = "data/raw_data/Admission_Predict_Ver1.1.csv"
    prepared_path = "data/admission.csv"

    feature_set = [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR ",
        "CGPA",
        "Research",
        "Chance of Admit ",
    ]
    if not os.path.exists(prepared_path):
        df = pd.read_csv(raw_path)
        df = df[feature_set]
        df["Label"] = df["Chance of Admit "].round(0).astype(int)
        df.rename(columns={"LOR ": "LOR"}, inplace=True)
        df = df.drop("Chance of Admit ", axis=1)
        df.to_csv(prepared_path, index=False)
    else:
        df = pd.read_csv(prepared_path)
    return df


def load_churn():
    raw_path = "data/raw_data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    prepared_path = "data/churn.csv"

    feature_set = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]
    if not os.path.exists(prepared_path):
        df = pd.read_csv(raw_path)
        df = df[feature_set]
        df["TotalCharges"].replace(" ", np.nan, inplace=True)
        df.dropna(subset=["TotalCharges"], inplace=True)
        df["TotalCharges"] = df["TotalCharges"].astype(float)
        df["Churn"] = np.where(df["Churn"] == "Yes", 1, 0)
        df.rename(columns={"Churn": "Label"}, inplace=True)
        df = pd.get_dummies(
            df,
            columns=[
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
            ],
            drop_first=True,
        )
        df.to_csv(prepared_path, index=False)
    else:
        df = pd.read_csv(prepared_path)
    return df


def load_fraud():
    raw_path = "data/raw_data/PS_20174392719_1491204439457_log.csv"
    prepared_path = "data/fraud.csv"

    feature_set = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]
    if not os.path.exists(prepared_path):
        df = pd.read_csv(raw_path)
        df = df[feature_set]
        df.rename(columns={"isFraud": "Label"}, inplace=True)
        df.to_csv(prepared_path, index=False)
    else:
        df = pd.read_csv(prepared_path)
    return df


def load_COMPAS():
    raw_path = "data/raw_data/compas-scores.csv"
    prepared_path = "data/COMPAS.csv"

    feature_set = [
        "sex",
        "age",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_jail_in",
        "c_jail_out",
        "c_charge_degree",
        "is_recid",
    ]
    if not os.path.exists(prepared_path):
        df = pd.read_csv(raw_path)
        df = df[feature_set]
        df["c_jail_in"] = pd.to_datetime(df["c_jail_in"])
        df.dropna(subset=["c_jail_in"], inplace=True)
        df["c_jail_out"] = pd.to_datetime(df["c_jail_out"])
        df.dropna(subset=["c_jail_out"], inplace=True)
        df["jailtime"] = (
            ((df["c_jail_out"] - df["c_jail_in"]) / np.timedelta64(1, "D"))
            .round(0)
            .astype(int)
        )
        df.drop(columns={"c_jail_in", "c_jail_out"}, inplace=True)
        df.rename(columns={"is_recid": "Label"}, inplace=True)
        df.dropna(subset=["Label"], inplace=True)
        df["Label"] = df["Label"].astype(int)
        df.to_csv(prepared_path, index=False)
    else:
        df = pd.read_csv(prepared_path)
    return df


def load_GTD():
    """Load and preprocess Global Terrorism Database (GTD).
    Features:
        - Numerical: nkill, nwound, nhostkid (casualties and hostages)
        - Categorical: attacktype1, weaptype1, targtype1, ransom
    Target: suicide (binary - whether attack was a suicide attack)

    Note: Only includes incidents from 1998 onwards (iyear > 1997)
    """
    raw_path = "data/raw_data/gtd.xlsx"
    prepared_path = "data/gtd.csv"

    numerical_features = ["nkill", "nwound", "nhostkid"]
    categorical_features = ["attacktype1", "weaptype1", "targtype1", "ransom"]
    target = "suicide"
    feature_set = numerical_features + categorical_features + [target, "iyear"]

    if not os.path.exists(prepared_path):
        df = pd.read_excel(raw_path)
        df = df[feature_set]
        df = df[df["iyear"] > 1997]
        df.drop(columns=["iyear"], inplace=True)
        df["nhostkid"] = df["nhostkid"].replace(-99, np.nan)
        df["ransom"] = df["ransom"].replace(-9, np.nan)
        df.dropna(subset=[target], inplace=True)
        df = df.sample(n=min(6000, len(df)), random_state=42)

        for cat_col in categorical_features:
            if cat_col in df.columns:

                df[cat_col] = df[cat_col].astype(str)
                dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[cat_col], inplace=True)

        for num_col in numerical_features:
            if num_col in df.columns:
                df[num_col].fillna(df[num_col].median(), inplace=True)
        df.rename(columns={target: "Label"}, inplace=True)
        df["Label"] = df["Label"].astype(int)
        df.to_csv(prepared_path, index=False)
    else:
        df = pd.read_csv(prepared_path)
    return df
