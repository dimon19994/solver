import numpy as np
from copy import deepcopy

from numpy.ma.core import equal

from main import display_plot, load_moments, load_sides
from scipy.interpolate import interp1d


def find_local_maxima(arr):
    half_max = max(arr) / 2
    maxima = []
    for i in range(1, len(arr) - 1):  # Не берем первый и последний элементы
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] > half_max:
            maxima.append(i)  # Сохраняем индекс локального максимума

    sorted_indices = sorted(maxima, key=lambda idx: arr[idx])
    return np.array(sorted(sorted_indices[:2]))


def find_local_minima(arr):
    half_min = min(arr) / 2
    minima = []
    for i in range(1, len(arr) - 1):  # Не берем первый и последний элементы
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1] and arr[i] < half_min:
            minima.append(i)  # Сохраняем индекс локального максимума

    sorted_indices = sorted(minima, key=lambda idx: arr[idx])
    return np.array(sorted(sorted_indices[:2]))

def compare_puzzles(p1_moments, p1_index, p1_side, p2_moments, p2_index, p2_side, move=True):
    s_1, s_2 = lead_to_one_size(p1_moments, p2_moments, move)

    if s_1 is None and s_2 is None:
        print("skip")

    if True:
        s_2_big = s_2

        if not move:
            s_1, s_2, qulity = get_best_quality_old(s_1, s_2_big)
            print("old", qulity)
        else:
            s_1, s_2, qulity = get_best_quality(s_1, s_2_big)
            print("new", qulity)
    else:
        s_2_part = s_2[:, 500: len(s_1[0]) + 500]
        M_j = np.array([s_1[0], s_2_part[1] - s_1[1]])

        m_j_dif = np.array([*[M_j[0][i + 1] - M_j[0][i] for i in range(M_j.shape[1] - 1)], M_j[0][-1] - M_j[0][-2]])
        M_j = np.vstack((M_j, m_j_dif))

        qulity = sum([M_j[1][ind] ** 2 * M_j[2][ind] for ind in range(len(M_j[1]))])



    print(f"{p1_index + 1}-{p1_side + 1}_{p2_index + 1}-{p2_side + 1} --> {qulity} {p1_moments[0][-1], p2_moments[0][-1]}")

    display_plot([
        [s_1, "lines", f"Кривизни {p1_index}-{p1_side + 1}", "#010BE6", {}, True],
        [s_2, "lines", f"Кривизни {p2_index}-{p2_side + 1}", "#E6B700", {}, True],
    ], title=f"{p1_index}-{p1_side + 1}_{p2_index}-{p2_side + 1} --> {qulity:.6}",
        filename=f"curve_comparing/{L}/manual/curve_{p1_index}-{p1_side + 1}_{p2_index}-{p2_side + 1}", html=True, s_json=True)


def get_best_quality(s_1, s_2):
    best_quality = 1000
    best_s_1 = []
    best_s_2 = []
    splits = 8

    start = 0
    amount = 1000
    step = round((amount / splits))

    while step >= 5:
        best_quality = 1000
        for i in range(splits + 1):
            step = (amount / splits)

            s_2_part = s_2[:, round(step * i) + start: len(s_1[0]) + round(step * i) + start]
            M_j = np.array([s_1[0], s_2_part[1] - s_1[1]])

            m_j_dif = np.array([*[M_j[0][i + 1] - M_j[0][i] for i in range(M_j.shape[1] - 1)], M_j[0][-1] - M_j[0][-2]])
            M_j = np.vstack((M_j, m_j_dif))

            qulity = sum([M_j[1][ind] ** 2 * M_j[2][ind] for ind in range(len(M_j[1]))])
            # print(qulity)

            if qulity < best_quality:
                new_start = round(step * i - step)
                new_amount = round(step * 2)
                best_quality = qulity
                best_s_1 = s_1
                best_s_2 = s_2_part

        start += 0 if new_start < 0 else new_start
        amount = amount - start if new_amount + new_start > amount else new_amount
        # print()

    return best_s_1, best_s_2, best_quality

def get_best_quality_old(s_1, s_2):
    best_quality = 1000
    best_s_1 = []
    best_s_2 = []

    M_j = np.array([s_1[0], s_2[1] - s_1[1]])

    m_j_dif = np.array([*[M_j[0][i + 1] - M_j[0][i] for i in range(M_j.shape[1] - 1)], M_j[0][-1] - M_j[0][-2]])
    M_j = np.vstack((M_j, m_j_dif))

    qulity = sum([M_j[1][ind] ** 2 * M_j[2][ind] for ind in range(len(M_j[1]))])
    # print(qulity)

    if qulity < best_quality:
        best_quality = qulity
        best_s_1 = s_1
        best_s_2 = s_2

    return best_s_1, best_s_2, best_quality


def lead_to_one_size(side_1, side_2, move=True):
    if side_1.shape[1] < side_2.shape[1]:
        min_side = deepcopy(side_1)
        max_side = deepcopy(side_2)
        save_order = True
    else:
        min_side = deepcopy(side_2)
        max_side = deepcopy(side_1)
        save_order = False
    min_side[0] = min_side[0] * (max_side[0][-1] / min_side[0][-1])

    new_min = np.empty((0, 2))
    for i in range(max_side.shape[1]):
        idx = np.searchsorted(min_side[0], max_side[0][i])
        if idx == 0:
            x1, y1 = min_side[:, 0]
            x2, y2 = min_side[:, 1]
        elif idx == len(min_side[0]):
            x1, y1 = min_side[:, -2]
            x2, y2 = min_side[:, -1]
        else:
            x1, y1 = min_side[:, idx - 1]
            x2, y2 = min_side[:, idx]

        coef = (max_side[0][i] - x1) / (x2 - x1)

        point = (x1 + coef * (x2 - x1), y1 + coef * (y2 - y1))
        new_min = np.append(new_min, [point], axis=0)

    if move:
        return (new_min.transpose()[:, 500:-500], max_side) if save_order else (max_side[:, 500:-500], new_min.transpose())
    else:
        return (new_min.transpose(), max_side) if save_order else (max_side, new_min.transpose())



def align_two_point_pairs(arr1, arr2, idx1=3000, idx2=-3000):
    # Векторы между контрольными точками
    vec1 = arr1[idx2] - arr1[idx1]
    vec2 = arr2[idx2] - arr2[idx1]

    # Расчёт угла поворота
    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])
    rotation_angle = angle1 - angle2

    # Поворот arr2 вокруг точки idx1
    def rotate(points, center, angle):
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        return (points - center) @ rot_matrix.T + center

    arr2_rotated = rotate(arr2, arr2[idx1], rotation_angle)

    # После поворота сдвигаем на вектор разницы между соответствующими точками
    shift = arr1[idx1] - arr2_rotated[idx1]
    arr2_aligned = arr2_rotated + shift

    return arr2_aligned.T

def align_two_point_pairs_same(arr1, arr2, idx1=3000, idx2=-3000):
    # Векторы между контрольными точками
    vec1 = arr1[idx2] - arr1[idx1]
    vec2 = arr2[-idx2] - arr2[-idx1]

    # Расчёт угла поворота
    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])
    rotation_angle = angle1 - angle2

    # Поворот arr2 вокруг точки idx1
    def rotate(points, center, angle):
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        return (points - center) @ rot_matrix.T + center

    arr2_rotated = rotate(arr2, arr2[idx1], rotation_angle)

    # После поворота сдвигаем на вектор разницы между соответствующими точками
    shift = arr1[idx1] - arr2_rotated[-idx1]
    arr2_aligned = arr2_rotated + shift

    return arr2_aligned.T


def process_loads(data, numbers):
    if len(data) == 2:
        p_s = {}
        p_r = {}
        for ind, i in enumerate(numbers):
            p_s[i] = data[0][ind]
            p_r[i] = data[1][ind]
        return p_s, p_r
    else:
        p = {}
        for ind, i in enumerate(numbers):
            p[i] = data[0][ind]
        return p



if __name__ == "__main__":
    pares = [

        # {
        #     2: 3,
        #     3: 3
        # },
        # {
        #     3: 3,
        #     4: 3
        # },
        # {
        #     5: 0,
        #     14: 2
        # },
        #
        # {
        #     1: 0,
        #     18: 2
        # },
        #
        # {
        #     9: 0,
        #     10: 2
        # },
        #
        # {
        #     8: 0,
        #     11: 2
        # },

        # {
        #     2: 3,
        #     54: 2,
        # },



        # {
        #     1: 3,
        #     2: 3
        # },
        # {
        #     1: 3,
        #     54: 2,
        # },



        # {
        #     23: 3,
        #     24: 3,
        # },
        #
        #
        # {
        #     21: 2,
        #     22: 0,
        # },
        # {
        #     27: 0,
        #     36: 2,
        # },

        {
            8: 2,
            9: 0,
        },
        # {
        #     8: 2,
        #     44: 1,
        # },

        # {
        #     5: 0,
        #     8: 2,
        # }
    ]


    Ls = ["5.0", "8.0", "10.0", "12.0", "15.0"]

    for L in Ls:
        puzzles_to_load = set([item for sublist in pares for item in sublist])

        moments_s, moments_r = process_loads(load_moments(puzzles_to_load, L), puzzles_to_load)
        puzzle_s, puzzle_r = process_loads(load_sides(puzzles_to_load, L), puzzles_to_load)

        for p in pares:
            first, second = p
            compare_puzzles(
                moments_r[first][p[first]], first, p[first],
                moments_s[second][p[second]], second, p[second],
                # move=False
            )

            bbbbb = align_two_point_pairs(puzzle_r[first][p[first]].T, puzzle_s[second][p[second]].T)
            # bbbbb = align_two_point_pairs_same(puzzle_r[first][p[first]].T, puzzle_s[second][p[second]].T)
            # bbbbb_s = align_two_point_pairs_same(puzzle_r[first][p[first]].T, puzzle_r[second][p[second]-2].T)

            display_plot([
                [puzzle_r[first][p[first]], "lines", f"Сторона {first}-{p[first] + 1}", "#010BE6", {}, True],
                [puzzle_s[second][p[second]], "lines", f"Сторона {second}-{p[second] + 1} not rot", "#E6B700", {}, True],
                [bbbbb, "lines", f"Сторона {second}-{p[second] + 1}", "#E6B700", {}, True],
                # [bbbbb_s, "lines", f"Сторона {second}-{p[second] + 1}_same", "#FF00FF", {}, True],

            ], title=f"{first}-{p[first] + 1}_{second}-{p[second] + 1}",
                filename=f"curve_comparing/{L}/manual/side_{first}-{p[first] + 1}_{second}-{p[second] + 1}",
                html=True, equal=True, s_json=True)


