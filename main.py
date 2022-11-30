from config import config_dict
from experiments import authoritativeness


def generate_row(mean, std, n_inconsistent_forcings, trivial, none, some, all):
    return f"\multicolumn{{1}}{{l|}}{{\\begin{{tabular}}[c]{{@{{}}l@{{}}}}$\mu={mean}$\\\\ $\\sigma={std}$\\\\ $N_{{inc}}={n_inconsistent_forcings}$\\\\ $N_{{tws}}={trivial}$\\\\ $N_{{none}}={none}$\\\\ $N_{{some}}={some}$\\\\ $N_{{all}}={all}$\end{{tabular}}}}"


def output_local(name, results):
    import os

    if not os.path.exists("results"):
        os.makedirs("results")
    with open(f"results/{name}.txt", "w") as f:
        print(f"Writing results for {name}.")
        f.write(results)


def evaluate(exp_dict, local=True):
    exp_dict = authoritativeness.experiment(exp_dict)
    for name in exp_dict:
        results = ""
        set = exp_dict[name].get("Full dataset")
        for auth_method in set["auth_methods"]:
            mean = set["auth_methods"][auth_method]["mean"]
            std = set["auth_methods"][auth_method]["std"]
            n_inconsistent_forcings = set["auth_methods"][auth_method][
                "Inconsistent forcings"
            ]
            trivial = set["auth_methods"][auth_method]["trivial"]
            none = set["auth_methods"][auth_method]["none"]
            some = set["auth_methods"][auth_method]["some"]
            all = set["auth_methods"][auth_method]["all"]
            row = generate_row(
                mean, std, n_inconsistent_forcings, trivial, none, some, all
            )
            results += row
            results += "&"
        if local:
            output_local(name, results)
        # else:
        # from AAP import kubernetes_s3
        # kubernetes_s3(name, results)


if __name__ == "__main__":
    evaluate(config_dict, local=True)
