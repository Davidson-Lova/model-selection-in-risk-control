def make_FNR_empirical_risk(outputs, probas):
    def FNR_empirical_risk(order):
        return (
            (np.logical_and(outputs, probas < 1 - order)).sum(axis=1)
            / outputs.sum(axis=1)
        ).mean()

    return FNR_empirical_risk

def make_FNR_lower_empirical_risk(outputs, probas):
    sample_size = outputs.shape[0]
    def FNR_lower_empirical_risk(order):
        tmp = (
            (np.logical_and(outputs, probas < 1 - order)).sum(axis=1)
            / outputs.sum(axis=1)
        ).sum()
        return (0.0 + tmp) / (sample_size + 1)

    return FNR_lower_empirical_risk

def make_FNR_upper_empirical_risk(outputs, probas):
    sample_size = outputs.shape[0]
    def FNR_upper_empirical_risk(order):
        tmp = (
            (np.logical_and(outputs, probas < 1 - order)).sum(axis=1)
            / outputs.sum(axis=1)
        ).sum()
        return (1.0 + tmp) / (sample_size + 1)

    return FNR_upper_empirical_risk

def size_function(probas, order):
    return (probas >= 1 - order).sum(axis=1).mean()

def compute_order(
    risk, risk_control_level, change_points
):
    index = dichotomy_search(
        lambda q: (risk(q) - risk_control_level), change_points
    )
    return change_points[index]