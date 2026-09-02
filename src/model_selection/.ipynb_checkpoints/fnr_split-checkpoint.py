def select_predictor_index(outputs_selection, probas_selection_per_predictor, risk_control_level):
    selection_change_points_per_predictor = [
        np.concatenate(
            (
                [0.0],
                np.sort(1 - probas_selection.flatten()),
                [1.0],
            )
        )
        for probas_selection in probas_selection_per_predictor
    ]

    selection_risk_per_predictor = [
        make_FNR_upper_empirical_risk(outputs_selection, probas_selection)
        for probas_selection in probas_selection_per_predictor
    ]
    selection_upper_order_per_predictor = [
        compute_order(
            risk, params_global["risk"]["control_level"], change_points
        )
        for risk, change_points in zip(
            selection_risk_per_predictor, selection_change_points_per_predictor
        )
    ]
    selection_upper_size_per_predictor = [
        size_function(probas_selection, order)
        for probas_selection, order in zip(
            probas_selection_per_predictor, selection_upper_order_per_predictor
        )
    ]

    predictor_index = np.argmin(selection_upper_size_per_predictor)
    return predictor_index