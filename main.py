import pandas as pd
from sklearn.model_selection import train_test_split

from classifier import kfold_random_forest
from config import config_dict
from experiments import authoritativeness
from preprocessing import get_data


def generate_table_cell(mean, n_inconsistent_forcings):
    if mean is None:
        return
    mean = round(mean, 2)
    return f"\multicolumn{{1}}{{l|}}{{\\begin{{tabular}}[c]{{@{{}}l@{{}}}}$\mu={mean}$\\\\ $N_{{inc}}={n_inconsistent_forcings}$\end{{tabular}}}}"


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
        exp_dict = authoritativeness.experiment(exp_dict)
        df_results = pd.DataFrame()
        latex_results = f"\multicolumn{{1}}{{l|}}{{{name}}} "
        set = exp_dict[name].get("Full dataset")
        for auth_method in set["auth_methods"]:
            latex_results += "&"
            mu = set["auth_methods"][auth_method].get("mean")
            Ninc = set["auth_methods"][auth_method]["Inconsistent forcings"]
            latex_cell = generate_table_cell(mu, Ninc)
            latex_results += latex_cell
            if auth_method.startswith("harmonic"):
                beta = float(auth_method.split("_")[1])
                data = {"beta": [beta], "mu": [mu], "Ninc": [Ninc]}
                df_results = df_results.append(pd.DataFrame(data).set_index("beta"))
        latex_results += " \\\\\cline{2-5}"
        output_local(name, latex_results, df_results)


def Q_evaluation(name):
    labelled_data = get_data(name)
    Q_total, metrics, CB_results = get_Q_with_preds(labelled_data)
    df_results = grow_Q(Q_total.head(1000))
    print("Found classifier with the following performance metrics: ", metrics)
    print("CB results: ", CB_results)
    output_local("Q_" + name, "", df_results)


def grow_Q(Q_total):
    Q = pd.DataFrame()
    df_results = pd.DataFrame()
    for _, row in Q_total.iterrows():
        Q = Q.append(row)
        results = authoritativeness.evaluate_dataset("", auth_method="default", df=Q)
        mu = results.get("mean")
        Ninc = results.get("Inconsistent forcings")
        data = {"|Q|": [len(Q)], "mu": [mu], "Ninc": [Ninc]}
        df_results = df_results.append(pd.DataFrame(data).set_index("|Q|"))
    return df_results


def get_Q_with_preds(data):
    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    X, Q, y, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    clf, metrics = kfold_random_forest(X, y)
    predicted_labels = clf.predict(Q)
    Q["Label"] = predicted_labels
    X["Label"] = y
    CB_results = authoritativeness.evaluate_dataset("", auth_method="default", df=X)
    return Q, metrics, CB_results


if __name__ == "__main__":
    # Q_evaluation("admission")
    evaluate_metrics(config_dict)
