import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import json
import requests

import numpy as np
from sklearn.linear_model import LinearRegression


def intersection(model1, model2, X1, X2):
    def is_vertical(X):
        return np.allclose(X[:, 0], X[0, 0])

    vertical1 = is_vertical(X1)
    vertical2 = is_vertical(X2)

    if vertical1 and vertical2:
        if np.isclose(X1[0, 0], X2[0, 0]):
            raise ValueError("Прямые совпадают (обе вертикальные и x равны)")
        else:
            raise ValueError("Прямые параллельны (обе вертикальные)")

    if vertical1:
        x = X1[0, 0]
        k2 = model2.coef_[0]
        b2 = model2.intercept_
        y = k2 * x + b2
        return np.array([x, y])

    if vertical2:
        x = X2[0, 0]
        k1 = model1.coef_[0]
        b1 = model1.intercept_
        y = k1 * x + b1
        return np.array([x, y])

    # обе обычные линии
    k1, b1 = model1.coef_[0], model1.intercept_
    k2, b2 = model2.coef_[0], model2.intercept_

    if np.isclose(k1, k2):
        raise ValueError("Прямые параллельны (наклоны равны)")

    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return np.array([x, y])


def send_req(array):
    arr_expanded = np.hstack((array, np.zeros((array.shape[0], 1))))
    data = {
        "data": json.dumps(arr_expanded.tolist()),
        "iter_count": 4,
        "start_value": 0.01,
        "end_value": 0.0005,
        "curve_type": "not_loop",
        # "curve_type": "loop",
        "save_data": 1,  # TODO change
        "file_name": "other",
        "straight": 1,
        "puzzle_index": -1,
        "general_l": 20,
        "side": -1,
        "show_new_plots": 0,
    }

    resp = requests.post("http://127.0.0.1:5254/calculate", data=data)

    try:
        smooth = np.array(json.loads(resp.json()["curve"]))
    except Exception as e:
        print("dedede", e)
    return smooth


def linear_fit(points):
    model = LinearRegression()
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model.fit(X, y)
    return model, X
#
# def intersection(model1, model2):
#     k1, b1 = model1.coef_[0], model1.intercept_
#     k2, b2 = model2.coef_[0], model2.intercept_
#     if np.isclose(k1, k2):
#         return None
#     x = (b2 - b1) / (k1 - k2)
#     y = k1 * x + b1
#     return np.array([x, y])

def interpolate_line(p1, p2, num=60):
    return np.linspace(p1, p2, num=num)

def get_circular_slice(start, end, n):
    """Возвращает список индексов от start до end по кругу"""
    if start < end:
        return list(range(start, end))
    else:
        return list(range(start, n)) + list(range(0, end))

def fix_corner_between_sides(contour, corner_idx, prev_corner, next_corner, offset=30, window=40):
    n = len(contour)
    get_idx = lambda i: i % n

    # Индексы от предыдущей до текущей и от текущей до следующей
    side1 = get_circular_slice(prev_corner, corner_idx, len(contour))
    side2 = get_circular_slice(corner_idx, next_corner, len(contour))

    # Точки отступа и окна
    # s1_start = side1[offset: offset + window] + side1[-(offset + window):-offset]
    # s2_start = side2[offset:offset + window] + side2[-(offset + window):-offset]
    s1_start = side1[-(offset + window):-offset]
    s2_start = side2[offset:offset + window]

    if len(s1_start) < window or len(s2_start) < window:
        return contour  # мало точек — пропускаем

    before = contour[s1_start][:, 0]
    after = contour[s2_start][:, 0]

    smooth_before = send_req(before)
    smooth_after = send_req(after)

    # plt.figure(figsize=(6, 6))
    # plt.plot(smooth_before[:, 0], smooth_before[:, 1], label="Smooth before", color='lightgray')
    # plt.plot(smooth_after[:, 0], smooth_after[:, 1], label="Smooth after", color='lightpink')
    # plt.scatter(before[:, 0], before[:, 1], color='blue', label="Before")
    # plt.scatter(after[:, 0], after[:, 1], color='green', label="After")
    # plt.title(f"Corner {corner_idx}")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    model1, X1 = linear_fit(smooth_before)
    model2, X2 = linear_fit(smooth_after)

    cross = intersection(model1, model2, X1, X2)
    if cross is None:
        return contour

    p1 = before[-1]
    p2 = after[0]
    interpolated_1 = interpolate_line(p1, cross, num=offset + 1)
    interpolated_1 = interpolated_1.reshape(-1, 1, 2)

    interpolated_2 = interpolate_line(cross, p2, num=offset + 1)
    interpolated_2 = interpolated_2.reshape(-1, 1, 2)

    interpolated = np.concatenate([interpolated_1, interpolated_2])

    # Индексы точек на замещение (30 до и 30 после текущего угла)
    replace_range = [get_idx(i) for i in range(corner_idx - offset, corner_idx + offset)]
    new_contour = contour.copy()
    new_contour = new_contour.astype(float)
    for i, idx in enumerate(replace_range):
        new_contour[idx, 0] = interpolated[i+1]

    # "\n".join(["{} {}".format(int(x), int(y)) for x, y in before])
    # x_line1 = np.linspace(before[0][0], cross[0], 100).reshape(-1, 1)
    # x_line1 = np.linspace(smooth_before[0][0], smooth_before[-1][0], 100).reshape(-1, 1)
    # y_line1 = model1.predict(x_line1)
    # x_line2 = np.linspace(smooth_after[0][0], smooth_after[-1][0], 100).reshape(-1, 1)
    # y_line2 = model2.predict(x_line2)

    # # Визуализация
    # plt.figure(figsize=(6, 6))
    # plt.plot(contour[:, 0, 0], contour[:, 0, 1], label="Original", color='lightgray')
    # plt.scatter(before[:, 0], before[:, 1], color='blue', label="Before")
    # plt.scatter(after[:, 0], after[:, 1], color='green', label="After")
    # plt.scatter(interpolated[:, 0, 0], interpolated[:, 0, 1], color='red', s=10, label="Interpolated")
    # plt.scatter([cross[0]], [cross[1]], color='black', marker='x', s=50, label="Intersection")
    # # plt.scatter(x_line1, y_line1, color='pink', label='Linear 1 fit', s=10)
    # # plt.scatter(x_line2, y_line2, color='pink', label='Linear 1 fit', s=10)
    # plt.title(f"Corner {corner_idx}")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return new_contour

def fix_all_between_corners(contour, corner_indices, offset=30, window=40):
    fixed = contour.copy()
    n = len(corner_indices)

    for i in range(n):
        curr = corner_indices[i]
        prev = corner_indices[(i - 1) % n]
        next = corner_indices[(i + 1) % n]
        fixed = fix_corner_between_sides(fixed, curr, prev, next, offset, window)

    # # Финальный график
    # plt.figure(figsize=(8, 8))
    # plt.plot(contour[:, 0, 0], contour[:, 0, 1], label="Original", linestyle='--', alpha=0.6)
    # plt.plot(fixed[:, 0, 0], fixed[:, 0, 1], label="Fixed", color='black')
    # plt.title("Final Contour After All Fixes")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return fixed


def split_contour_by_corners(contour, corner_indices):
    n = len(contour)
    contour_parts = []

    for i in range(len(corner_indices)):
        start = corner_indices[i]
        end = corner_indices[(i + 1) % len(corner_indices)]
        idxs = get_circular_slice(start, end, n)

        # (M, 2) — 2D массив
        part = contour[idxs, 0]  # из (M, 1, 2) → (M, 2)
        contour_parts.append(part)

    return contour_parts



# for i in range(5, 6):
#     filename = f"all_good_{i}.txt"
#     print(filename)
#
#     arr = np.loadtxt(f"../PDF/res_data/{filename}").T
#
#     contour = np.stack(arr, axis=1).astype(np.int32).reshape(-1, 1, 2)[::-1]
#     corners = [9, 645, 1126, 1722]
#
#     fixed_contour = fix_all_between_corners(contour, corner_indices=corners, offset=25, window=50)
#
#     split_contour_by_corners(fixed_contour, corner_indices=corners)


