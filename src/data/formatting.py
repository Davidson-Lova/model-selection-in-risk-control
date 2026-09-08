import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def resample_data_set(inputs_, outputs_, cal_size, proper_cal_size):
    sample_size = inputs_.shape[0] - 1
    indices = np.arange(sample_size + 1)
    np.random.shuffle(indices)

    inputs_ = inputs_[indices, :]
    outputs_ = outputs_[indices, :]

    inputs, outputs = (inputs_[:-1, :], outputs_[:-1, :])
    input_test, output_test = (
        inputs_[-1, :].reshape(1, -1),
        outputs_[-1, :].reshape(1, -1),
    )

    inputs_train, inputs_calibration, outputs_train, outputs_calibration = (
        train_test_split(inputs, outputs, test_size=cal_size)
    )

    input_scaler = StandardScaler()
    scaled_inputs_train = input_scaler.fit_transform(inputs_train)
    scaled_inputs_calibration = input_scaler.transform(inputs_calibration)
    scaled_input_test = input_scaler.transform(input_test)

    (
        scaled_inputs_selection,
        scaled_inputs_proper_cal,
        outputs_selection,
        outputs_proper_cal,
    ) = train_test_split(
        scaled_inputs_calibration,
        outputs_calibration,
        test_size=proper_cal_size,
    )

    return (
        (scaled_inputs_train, outputs_train),
        (scaled_inputs_calibration, outputs_calibration),
        (
            scaled_inputs_selection,
            scaled_inputs_proper_cal,
            outputs_selection,
            outputs_proper_cal,
        ),
        (scaled_input_test, output_test),
    )