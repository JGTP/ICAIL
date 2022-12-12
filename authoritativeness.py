def alpha(case, CB, method):
    if method == "relative":
        return relative_authoritativeness(case, CB)
    elif method == "absolute":
        return absolute_authoritativeness(case, CB)
    elif method == "product":
        return product_authoritativeness(case, CB)
    elif method.startswith("harmonic"):
        beta = float(method.split("_")[1])
        return harmonic_authoritativeness(case, CB, beta)
    else:
        raise ValueError("Unknown method for authoritativeness.")


def n_agreement(case, CB):
    same_outcome = [c for c in CB if c.s == case.s]
    precedents = []
    for other_case in same_outcome:
        if case <= other_case:
            precedents.append(other_case.name)
    return len(precedents)


def n_disagreement(case, CB):
    other_outcome = [c for c in CB if c.s != case.s]
    precedents = []
    for other_case in other_outcome:
        if case <= other_case:
            precedents.append(other_case.name)
    return len(precedents)


def relative_authoritativeness(case, CB):
    n_a = n_agreement(case, CB)
    n_d = n_disagreement(case, CB)
    return n_a / (n_a + n_d)


def absolute_authoritativeness(case, CB):
    n_a = n_agreement(case, CB)
    return n_a / len(CB)


def product_authoritativeness(case, CB):
    rels = relative_authoritativeness(case, CB)
    abss = absolute_authoritativeness(case, CB)
    return rels * abss


def harmonic_authoritativeness(case, CB, beta):
    rels = relative_authoritativeness(case, CB)
    abss = absolute_authoritativeness(case, CB)
    return (1 + beta**2) * (rels * abss) / ((beta**2 * rels) + abss)
