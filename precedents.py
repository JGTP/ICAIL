import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def determine_distribution(n_precedents):
    n_precedents = np.array(n_precedents)
    mean = round(n_precedents.mean(), 2)
    std = round(n_precedents.std(), 2)
    return mean, std


def _process_single_case(case, CB):
    """Process a single case to get precedent statistics."""
    best_precedents = get_best_precedents(case, CB)
    n_prec = len(best_precedents)

    # Determine if non-trivial
    is_nontrivial = not has_trivial_winning_strategy(best_precedents)

    # Count strategy types
    strategy_counts = {"all": 0, "some": 0, "none": 0, "trivial": 0}
    if not has_trivial_winning_strategy(best_precedents):
        n_empties = sum(1 for p in best_precedents if p["requires_empty"])
        if n_empties == 0:
            strategy_counts["none"] = 1
        elif n_empties == len(best_precedents):
            strategy_counts["all"] = 1
        else:
            strategy_counts["some"] = 1
    else:
        strategy_counts["trivial"] = 1

    return n_prec, is_nontrivial, strategy_counts


def get_precedent_distribution(CB, n_jobs=-1):
    """
    Compute precedent distribution with optional parallel processing.

    Args:
        CB: Case base
        n_jobs: Number of parallel jobs (-1 = use all cores, 1 = no parallelisation)
    """
    # Process all cases in parallel
    case_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_case)(case, CB)
        for case in tqdm(CB, desc="Computing precedent distribution")
    )

    # Aggregate results
    n_precedents = []
    n_precedents_nontrivial = []
    results = {"all": 0, "some": 0, "none": 0, "trivial": 0}

    for n_prec, is_nontrivial, strategy_counts in case_results:
        n_precedents.append(n_prec)
        if is_nontrivial:
            n_precedents_nontrivial.append(n_prec)

        for key in strategy_counts:
            results[key] += strategy_counts[key]

    # Compute statistics
    results["mean"], results["std"] = determine_distribution(n_precedents)

    if n_precedents_nontrivial:
        results["mean_nontrivial"], _ = determine_distribution(n_precedents_nontrivial)
    else:
        results["mean_nontrivial"] = None

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
    if not comparisons:
        return []

    # Pre-compute trivial/requires_empty for all comparisons
    for c in comparisons:
        c["trivial"], c["requires_empty"] = determine_trivial_or_requires_empty(c)

    bested = set()
    use_alpha = CB.auth_method != "default"

    for c in comparisons:
        if c["name"] in bested:
            continue

        c_diffs = c["rel_differences"]
        for oc in comparisons:
            if oc["name"] == c["name"] or oc["name"] in bested:
                continue

            # Check if c is worse than oc
            if c_diffs > oc["rel_differences"]:
                if not use_alpha or c["alpha"] <= oc["alpha"]:
                    bested.add(c["name"])
                    break  # Early termination - c is already bested

    return [c for c in comparisons if c["name"] not in bested]


def determine_trivial_or_requires_empty(c):
    if len(c["rel_differences"]) > 0 and len(c["comp_differences"]) == 0:
        return False, True
    elif len(c["rel_differences"]) == 0:
        return True, False
    else:
        return False, False


def get_comparisons(f, CB):
    if CB.auth_method == "default":
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
