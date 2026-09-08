import numpy as np
from ..utils.search import dichotomy_search
from scipy.special import softmax


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


def compute_order(risk, risk_control_level, change_points):
    index = dichotomy_search(lambda q: (risk(q) - risk_control_level), change_points)
    return change_points[index]


def compute_prediction_set(
    predictor,
    inputs_calibration,
    output_calibrations,
    input_test,
    risk_control_level,
    apply_softmax=False,
):
    if apply_softmax:
        probas_calibration = softmax(predictor.predict_proba(inputs_calibration))
    else:
        probas_calibration = predictor.predict_proba(inputs_calibration)

    change_points = np.concatenate(
        (
            [0.0],
            np.sort(1 - probas_calibration.flatten()),
            [1.0],
        )
    )

    upper_risk = make_FNR_upper_empirical_risk(output_calibrations, probas_calibration)
    upper_order = compute_order(upper_risk, risk_control_level, change_points)

    proba_test = predictor.predict_proba(input_test)
    prediction_set = np.int64(proba_test >= 1 - upper_order)

    return prediction_set
