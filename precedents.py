from unittest import result

import numpy as np
from tqdm import tqdm


def determine_distribution(n_precedents):
    n_precedents = np.array(n_precedents)
    mean = round(n_precedents.mean(), 2)
    std = round(n_precedents.std(), 2)
    return mean, std


def get_precedent_distribution(CB):
    n_precedents = []
    results = {"all": 0, "some": 0, "none": 0, "trivial": 0}
    for case in tqdm(CB):
        best_precedents = get_best_precedents(case, CB)
        n_precedents.append(len(best_precedents))
        results = determine_strategy_counts(results, best_precedents)
    results["mean"], results["std"] = determine_distribution(n_precedents)
    return results


def determine_strategy_counts(results, best_precedents):
    if not has_trivial_winning_strategy(best_precedents):
        n_empties = len([p for p in best_precedents if p["requires_empty"] is True])
        if n_empties == 0:
            results["none"] += 1
        elif n_empties == len(best_precedents):
            results["all"] += 1
        else:
            results["some"] += 1
    else:
        results["trivial"] += 1
    return results


def has_trivial_winning_strategy(best_precedents):
    trivials = [p for p in best_precedents if p["trivial"] is True]
    return len(trivials) > 0


def get_best_precedents(f, CB):
    comparisons = get_comparisons(f, CB)
    bested = set()
    for c in comparisons:
        c["trivial"], c["requires_empty"] = determine_trivial_or_requires_empty(c)
        if c["name"] not in bested:
            if CB.auth_method is None:
                bested = inner_loop_naive(comparisons, c, bested)
            else:
                bested = inner_loop_alpha(comparisons, c, bested)
    return [c for c in comparisons if c["name"] not in bested]


def determine_trivial_or_requires_empty(c):
    if len(c["rel_differences"]) > 0 and len(c["comp_differences"]) == 0:
        return False, True
    elif len(c["rel_differences"]) == 0:
        return True, False
    else:
        return False, False


def get_comparisons(f, CB):
    if CB.auth_method is None:
        return [
            {
                "name": c.name,
                "rel_differences": set(c.diff(f.F)),
                "comp_differences": set(c.comp_diff(f.F)),
            }
            for c in CB
            if c.s == f.s and c.name != f.name
        ]
    else:
        return [
            {
                "name": c.name,
                "rel_differences": set(c.diff(f.F)),
                "comp_differences": set(c.comp_diff(f.F)),
                "alpha": c.alpha,
            }
            for c in CB
            if c.s == f.s and c.name != f.name
        ]


def inner_loop_naive(comparisons, c, bested):
    for oc in comparisons:
        if c["rel_differences"] > oc["rel_differences"]:
            bested.add(c["name"])
    return bested


def inner_loop_alpha(comparisons, c, bested):
    for oc in comparisons:
        if c["rel_differences"] > oc["rel_differences"] and c["alpha"] <= oc["alpha"]:
            bested.add(c["name"])
    return bested
