import preprocessing


def test_admission():
    df = preprocessing.get_data("admission")
    assert df.columns.__len__() == 8
    for c in df.columns:
        assert c in [
            "GRE Score",
            "TOEFL Score",
            "University Rating",
            "SOP",
            "LOR",
            "CGPA",
            "Research",
            "Label",
        ]
    assert df.iloc[0]["GRE Score"] == 337.00
    assert df.iloc[0]["CGPA"] == 9.65
    assert df.iloc[0]["Label"] == 1
    assert len(df) == 500


def test_churn():
    df = preprocessing.get_data("churn")
    assert df.columns.__len__() == 20
    for c in df.columns:
        assert c in [
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
            "Label",
        ]
    assert df.iloc[0]["gender"] == "Female"
    assert df.iloc[0]["TotalCharges"] == 29.85
    assert df.iloc[0]["Label"] == 0
    assert len(df) == 7032


def test_fraud():
    df = preprocessing.get_data("fraud")
    assert df.columns.__len__() == 8
    for c in df.columns:
        assert c in [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "Label",
        ]
    assert df.iloc[0]["type"] == "PAYMENT"
    assert df.iloc[0]["amount"] == 9839.64
    assert df.iloc[0]["Label"] == 0
    assert len(df) == 6362620


def test_COMPAS():
    df = preprocessing.get_data("COMPAS")
    assert df.columns.__len__() == 10
    for c in df.columns:
        assert c in [
            "sex",
            "age",
            "race",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
            "jailtime",
            "c_charge_degree",
            "Label",
        ]
    assert df.iloc[0]["sex"] == "Male"
    assert df.iloc[0]["jailtime"] == 1
    assert df.iloc[0]["Label"] == 0
    assert len(df) == 10577


def test_gtd():
    df = preprocessing.get_data("gtd")
    assert len(df) > 10
