def dichotomy_search(func, change_points):
    cardinal = change_points.shape[0]
    left_index = 0
    right_index = cardinal - 1

    f_right = func(change_points[right_index])
    f_left = func(change_points[left_index])

    if f_right * f_left > 0:
        if f_left > 0:
            return right_index
        else:
            return left_index

    while (right_index - left_index) >= 2:
        middle_index = max(
            min(np.int64((left_index + right_index) / 2), right_index), left_index
        )
        f_middle = func(change_points[middle_index])

        if (f_middle == 0) or (f_left * f_middle < 0):
            right_index = middle_index
            f_right = f_middle
        else:
            left_index = middle_index
            f_left = f_middle

    if f_left * f_right < 0:
        return right_index
    else:
        return left_index