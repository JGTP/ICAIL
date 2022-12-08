import pandas as pd

from config import config_dict
from experiments import authoritativeness


def generate_table_cell(mean, n_inconsistent_forcings):
    if mean is None:
        return
    mean = round(mean, 2)
    return f"\multicolumn{{1}}{{l|}}{{\\begin{{tabular}}[c]{{@{{}}l@{{}}}}$\mu={mean}$\\\\ $N_{{inc}}={n_inconsistent_forcings}$\end{{tabular}}}}"


def output_local(name, latex, df):
    import os

    if not os.path.exists("results"):
        os.makedirs("results")
    if len(latex) > 50:
        with open(f"results/{name}.txt", "w") as f:
            print(f"Writing results for {name}.")
            f.write(latex)
    if len(df) > 2:
        df.to_csv(f"results/{name}.csv", sep=";")


def evaluate(exp_dict, local=True):
    for name in exp_dict:
        exp_dict = authoritativeness.experiment(exp_dict)
        df_results = pd.DataFrame()
        latex_results = f"\multicolumn{{1}}{{l|}}{{{name}}} "
        set = exp_dict[name].get("Full dataset")
        for auth_method in set["auth_methods"]:
            latex_results += "&"
            mu = set["auth_methods"][auth_method].get("mean")
            Ninc = set["auth_methods"][auth_method]["Inconsistent forcings"]
            # std = set["auth_methods"][auth_method]["std"]
            # trivial = set["auth_methods"][auth_method]["trivial"]
            # none = set["auth_methods"][auth_method]["default"]
            # some = set["auth_methods"][auth_method]["some"]
            # all = set["auth_methods"][auth_method]["all"]
            if not auth_method.startswith("harmonic"):
                latex_cell = generate_table_cell(mu, Ninc)
                latex_results += latex_cell
            else:
                beta = float(auth_method.split("_")[1])
                data = {"beta": [beta], "mu": [mu], "Ninc": [Ninc]}
                df_results = df_results.append(pd.DataFrame(data).set_index("beta"))
        latex_results += " \\\\\cline{2-5}"
        if local:
            output_local(name, latex_results, df_results)
        # else:
        # from AAP import kubernetes_s3
        # kubernetes_s3(name, results)


if __name__ == "__main__":
    evaluate(config_dict, local=True)
