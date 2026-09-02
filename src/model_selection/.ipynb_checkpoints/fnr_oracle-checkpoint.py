def compute_oracle_prediction_set(
    outputs_calibration,
    probas_calibration_per_predictor,
    output_test,
    proba_test_per_predictor,
    risk_control_level,
):
    oracle_risk_per_predictor = [
        make_FNR_empirical_risk(
            np.concatenate((outputs_calibration, output_test)),
            np.concatenate((probas_calibration, proba_test)),
        )
        for probas_calibration, proba_test in zip(
            probas_calibration_per_predictor, proba_test_per_predictor
        )
    ]
    oracle_change_points_per_predictor = [
        np.concatenate(
            (
                [0.0],
                np.sort(1 - (np.concatenate((probas_calibration, proba_test))).flatten()),
                [1.0],
            )
        )
        for probas_calibration, proba_test in zip(
            probas_calibration_per_predictor, proba_test_per_predictor
        )
    ]
    oracle_order_per_predictor = [
        compute_order(
            risk, risk_control_level, change_points
        )
        for risk, change_points in zip(
            oracle_risk_per_predictor, oracle_change_points_per_predictor
        )
    ]
    oracle_size_per_predictor = [
        size_function(np.concatenate((probas_calibration, proba_test)), order)
        for probas_calibration, proba_test, order in zip(
            probas_calibration_per_predictor,
            proba_test_per_predictor,
            oracle_order_per_predictor,
        )
    ]

    oracle_predictor_index = np.argmin(oracle_size_per_predictor)
    oracle_prediction_set = np.int64(
        proba_test_per_predictor[oracle_predictor_index]
        >= 1 - oracle_order_per_predictor[oracle_predictor_index]
    )
    return oracle_prediction_set