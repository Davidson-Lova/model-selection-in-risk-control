def compute_upper_prediction_set(
    outputs_calibration,
    probas_calibration_per_predictor,
    proba_test_per_predictor,
    risk_control_level,
):
    change_points_per_predictor = [
        np.concatenate(
            (
                [0.0],
                np.sort(1 - probas_calibration.flatten()),
                [1.0],
            )
        )
        for probas_calibration in probas_calibration_per_predictor
    ]
 
    lower_risk_per_predictor = [
        make_FNR_lower_empirical_risk(outputs_calibration, probas_calibration)
        for probas_calibration in probas_calibration_per_predictor
    ]
    lower_order_per_predictor = [
        compute_order(
            risk, params_global["risk"]["control_level"], change_points
        )
        for risk, change_points in zip(lower_risk_per_predictor, change_points_per_predictor)
    ]
    lower_size_per_predictor = [
        size_function(np.concatenate((probas_calibration, proba_test)), order)
        for probas_calibration, proba_test, order in zip(
            probas_calibration_per_predictor,
            proba_test_per_predictor,
            lower_order_per_predictor,
        )
    ]

    upper_risk_per_predictor = [
        make_FNR_upper_empirical_risk(outputs_calibration, probas_calibration)
        for probas_calibration in probas_calibration_per_predictor
    ]
    upper_order_per_predictor = [
        compute_order(
            risk, params_global["risk"]["control_level"], change_points
        )
        for risk, change_points in zip(upper_risk_per_predictor, change_points_per_predictor)
    ]
    upper_size_per_predictor = [
        size_function(np.concatenate((probas_calibration, proba_test)), order)
        for probas_calibration, proba_test, order in zip(
            probas_calibration_per_predictor,
            proba_test_per_predictor,
            upper_order_per_predictor,
        )
    ]

    min_upper_size = np.min(upper_size_per_predictor)
    predictor_index_set = np.arange(lam_grid.shape[0])[
        lower_size_per_predictor <= min_upper_size
    ]
    upper_prediction_set = np.int64(
        [
            np.min(
                [
                    size_function(
                        np.concatenate(
                            (
                                probas_calibration_per_predictor[predictor_index],
                                proba_test_per_predictor[predictor_index],
                            )
                        ),
                        1 - proba_test_per_predictor[predictor_index][0, k],
                    )
                    for predictor_index in predictor_index_set
                ]
            )
            <= min_upper_size
            for k in range(params_global["data"]["label_number"])
        ]
    ).reshape(1, -1)

    return upper_prediction_set