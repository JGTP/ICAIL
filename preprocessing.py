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
    # elif dataset_name == "SCOTUS":
    #     df = load_SCOTUS()
    elif dataset_name == "COMPAS":
        df = load_COMPAS()
    else:
        raise ValueError("Unknown dataset")
    return df


# url: https://www.kaggle.com/datasets/mohansacharya/graduate-admissions
def load_admission():
    raw_path = "data/raw_data/Admission_Predict_Ver1.1.csv"
    prepared_path = "data//admission.csv"
    # Ignored feature(s): Serial No.
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


# url: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
def load_churn():
    raw_path = "data/raw_data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    prepared_path = "data//churn.csv"
    # Ignored feature(s): customerID
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
        df.to_csv(prepared_path, index=False)
    else:
        df = pd.read_csv(prepared_path)
    return df


# url: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
def load_fraud():
    raw_path = "data/raw_data/PS_20174392719_1491204439457_log.csv"
    prepared_path = "data//fraud.csv"
    # Ignored feature(s): nameOrig, nameDest, isFlaggedFraud
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


# url: https://github.com/propublica/compas-analysis
def load_COMPAS():
    raw_path = "data/raw_data/compas-scores.csv"
    prepared_path = "data//COMPAS.csv"
    # Ignored feature(s):
    #     id,
    #     name,
    #     first,
    #     last,
    #     compas_screening_date,
    #     dob,
    #     age_cat,
    #     num_r_cases,
    #     decile_score,
    #     days_b_screening_arrest,
    #     c_case_number,
    #     c_offense_date,
    #     c_arrest_date,
    #     c_days_from_compas,
    #     c_charge_desc,
    #     r_case_number,
    #     r_charge_degree,
    #     r_days_from_arrest,
    #     r_offense_date,
    #     r_charge_desc,
    #     r_jail_in,
    #     r_jail_out,
    #     is_violent_recid,
    #     num_vr_cases,
    #     vr_case_number,
    #     vr_charge_degree,
    #     vr_offense_date,
    #     vr_charge_desc,
    #     v_type_of_assessment,
    #     v_decile_score,
    #     v_score_text,
    #     v_screening_date,
    #     type_of_assessment,
    #     decile_score,
    #     score_text,
    #     screening_date,
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
