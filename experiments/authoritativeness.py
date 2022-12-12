import pandas as pd

from case_base import CaseBase
from precedents import get_precedent_distribution


def experiment(exp_dict):
    m = "pearson"
    for name in exp_dict:
        path = exp_dict[name]["path"]
        full_experiments = exp_dict[name].get("Full dataset")
        if full_experiments:
            make_consistent = False
            auth_methods = full_experiments["auth_methods"]
            print("\n===========================================")
            print(
                f"Analysing {name} using the {m} method with make_consistent={make_consistent}."
            )
            for auth_method in auth_methods:
                full_experiments["auth_methods"][auth_method] = evaluate_dataset(
                    path, auth_method, make_consistent, m
                )

        consistent_experiments = exp_dict[name].get("Consistent subset")
        if consistent_experiments:
            make_consistent = True
            auth_methods = consistent_experiments["auth_methods"]
            print("\n===========================================")
            print(
                f"Analysing {name} using the {m} method with make_consistent={make_consistent}."
            )
            for auth_method in auth_methods:
                consistent_experiments["auth_methods"][auth_method] = evaluate_dataset(
                    path, auth_method, make_consistent, m
                )
    return exp_dict


def evaluate_dataset(
    path, auth_method, make_consistent=False, m="pearson", max_size=8000, df=None
):
    results = {}
    print(f"\nEvaluating for auth_method={auth_method}...")
    if df is None:
        df = pd.read_csv(path)
    # df = df.head(10)
    CB = CaseBase(df, verb=True, method=m, auth_method=auth_method)
    if make_consistent:
        initial_size = len(CB)
        print(f"Initial size: {initial_size}.")
        CB.take_consistent_subset()
        reduced_size = len(CB)
        print(f"Reduced size: {reduced_size}.")
        print(
            f"Removed {initial_size - reduced_size} ({100*(initial_size - reduced_size)/initial_size} %)."
        )
    if len(CB) <= max_size:
        results = get_precedent_distribution(CB)
    else:
        print("Skipping mu due to large CB.")
    inds = range(len(CB))
    forcings = CB.get_forcings(inds, make_consistent)
    Id = CB.determine_inconsistent_forcings(inds, forcings)
    results["Inconsistent forcings"] = CB.get_n_inconst_forcings(Id)
    return results
