import pandas as pd
from sklearn.model_selection import train_test_split
from classifier import kfold_random_forest
from experiments import authoritativeness
from preprocessing import get_data


def generate_table_cell(mean, mean_nontrivial, n_inconsistent_forcings, n_del):
    if mean is None:
        return
    mean = round(mean, 2)
    mean_nt = round(mean_nontrivial, 2) if mean_nontrivial is not None else "N/A"
    return f"\\multicolumn{{1}}{{l|}}{{\\begin{{tabular}}[c]{{@{{}}l@{{}}}}$\\mu={mean}$\\\\ $\\mu_n={mean_nt}$\\\\ $N_{{inc}}={n_inconsistent_forcings}$\\\\ $N_{{del}}={n_del}$\\end{{tabular}}}}"


def output_local(name, latex, df):
    import os

    if not os.path.exists("results"):
        os.makedirs("results")
    with open(f"results/{name}.txt", "w") as f:
        print(f"Writing results for {name}.")
        f.write(latex)
    if len(df) > 2:
        df.to_csv(f"results/{name}.csv", sep=";")


def evaluate_metrics(exp_dict):
    for name in exp_dict:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {name}")
        print(f"{'='*50}")

        # Process this single dataset
        dataset_results = authoritativeness.experiment_single_dataset(
            name, exp_dict[name]
        )

        # Generate output immediately
        df_results = pd.DataFrame()
        latex_results = f"\\multicolumn{{1}}{{l|}}{{{name}}} "

        set = dataset_results.get("Full dataset")
        if set:
            for auth_method in set["auth_methods"]:
                latex_results += "&"
                mu = set["auth_methods"][auth_method].get("mean")
                mu_n = set["auth_methods"][auth_method].get("mean_nontrivial")
                Ninc = set["auth_methods"][auth_method]["Inconsistent forcings"]
                Ndel = set["auth_methods"][auth_method].get("N_del", 0)
                latex_cell = generate_table_cell(mu, mu_n, Ninc, Ndel)
                latex_results += latex_cell
                if auth_method.startswith("harmonic"):
                    beta = float(auth_method.split("_")[1])
                    data = {
                        "beta": [beta],
                        "mu": [mu],
                        "mu_n": [mu_n],
                        "Ninc": [Ninc],
                        "Ndel": [Ndel],
                    }
                    df_results = df_results.append(pd.DataFrame(data).set_index("beta"))
            latex_results += " \\\\\\cline{2-5}"

        # Write output immediately
        output_local(name, latex_results, df_results)
        print(f"? Results written for {name}")


def grow_Q(Q_total, auth_method="default"):
    Q = pd.DataFrame()
    df_results = pd.DataFrame()
    for _, row in Q_total.iterrows():
        Q = Q.append(row)
        results = authoritativeness.evaluate_dataset("", auth_method=auth_method, df=Q)

        n_n = results.get("all", 0) + results.get("some", 0) + results.get("none", 0)

        data = {
            "|Q|": [len(Q)],
            "mu": [results.get("mean")],
            "mu_n": [results.get("mean_nontrivial")],
            "Ninc": [results.get("Inconsistent forcings")],
            "Ndel": [results.get("N_del", 0)],
            "Ntws": [results.get("trivial", 0)],
            "Nn": [n_n],
        }
        df_results = df_results.append(pd.DataFrame(data).set_index("|Q|"))
    return df_results


def Q_evaluation(name):
    labelled_data = get_data(name)

    for auth_method in ["default", "harmonic_1"]:
        Q_total, metrics, CB_results = get_Q_with_preds(labelled_data, auth_method)
        df = grow_Q(Q_total, auth_method=auth_method)
        print("Found classifier with the following performance metrics: ", metrics)
        print("CB results: ", CB_results)
        output_local(f"Q_{name}_{auth_method}", "", df)


def get_Q_with_preds(data, auth_method):
    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    # X = pd.get_dummies(
    #     X,
    #     columns=[
    #         "gender",
    #         "Partner",
    #         "Dependents",
    #         "tenure",
    #         "PhoneService",
    #         "MultipleLines",
    #         "InternetService",
    #         "OnlineSecurity",
    #         "OnlineBackup",
    #         "DeviceProtection",
    #         "TechSupport",
    #         "StreamingTV",
    #         "StreamingMovies",
    #         "Contract",
    #         "PaperlessBilling",
    #         "PaymentMethod",
    #     ],
    #     drop_first=True,
    # )
    X, Q, y, _ = train_test_split(X, y, test_size=0.5, random_state=0)
    clf, metrics = kfold_random_forest(X, y)
    predicted_labels = clf.predict(Q)
    Q["Label"] = predicted_labels
    X["Label"] = y
    CB_results = authoritativeness.evaluate_dataset("", auth_method=auth_method, df=X)
    return Q, metrics, CB_results


if __name__ == "__main__":
    # evaluate_metrics(config_dict)
    Q_evaluation("churn")
