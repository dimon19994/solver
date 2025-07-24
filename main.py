from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from pyautogui import hotkey
import plotly.graph_objects as go
# from puzzle_comation import find_local_minima, find_local_maxima

import sys
import os
sys.path.append(os.path.abspath("../plots_storage"))
from plots import display_plot


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

def read_data_from_files(index_image, direction, L, type_="puzzle"):
    puzzle_sides = []
    for index_side in range(4):
        filename = f"all_good_{index_image}_{index_side + 1}_data"
        f = open(f"./../diploma_web/processed_pazzle_data/{L}/{type_}/{direction}/{filename}.txt")
        data = f.read().split("\n")
        f.close()
        plot_data = np.array([list(map(float, i.split(" "))) for i in data if i != ""])  # [20:-20])
        puzzle_sides.append(plot_data.transpose())

    return puzzle_sides


def load_moments(puzzles_len, L):
    puzzles = []
    puzzles_reverse = []
    for num, index_image in enumerate(puzzles_len):
    # for num, index_image in enumerate([1, 2, 3, 16, 17]):
        puzzles.append(read_data_from_files(index_image, "straight", L))
        puzzles_reverse.append(read_data_from_files(index_image, "reverse", L))

    return puzzles, puzzles_reverse

def load_sides(puzzles_len, L):
    puzzle_sides = []
    puzzle_sides_reverse = []
    for num, index_image in enumerate(puzzles_len):
    # for num, index_image in enumerate([1, 2, 3, 16, 17]):
        puzzle_sides.append(read_data_from_files(index_image, "straight", L,"conturs"))
        puzzle_sides_reverse.append(read_data_from_files(index_image, "reverse", L,"conturs"))

    return puzzle_sides, puzzle_sides_reverse

def lead_to_one_size(side_1, side_2):
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

    return (new_min.transpose()[:, 2500:-2500], max_side[:, 2000:-2000]) if save_order else (max_side[:, 2500:-2500], new_min.transpose()[:, 2000:-2000])


def get_side_quality(side):
    quality = 0
    for point in range(len(side[1])):
        quality += side[1][point] ** 2 * side[0][point]

    return quality


def find_first_puzzle(puzzles):
    min_diff = float('inf')
    best_match = None
    for j, first_puzzle in enumerate(puzzles):
        for k in range(4):
            M_j_1 = first_puzzle[k]
            M_j_2 = first_puzzle[(k+1)%4]

            qulity_1 = get_side_quality(M_j_1[:, 1000:-1000])
            qulity_2 = get_side_quality(M_j_2[:, 1000:-1000])

            if qulity_1 + qulity_2 < min_diff:
                best_match = [j, k]
                min_diff = qulity_1 + qulity_2

    print(f"Min side quality: {min_diff}")
    return best_match


def move_sides(puzzle, contur, shift):
    puzzle = puzzle[shift:] + puzzle[:shift]
    contur = contur[shift:] + contur[:shift]

    return puzzle, contur


def find_next_puzzle(puzzle_index, puzzles_reverse, main_side, main_index):
    min_diff = float('inf')
    best_match = None
    match_data = None
    for j, other_puzzle in enumerate(puzzles_reverse):
        if puzzle_index == j:
            continue
        for k, other_side in enumerate(other_puzzle):

            # With scale
            a, b = len(main_side[0]), len(other_side[0])
            if not (abs(a - b) / max(abs(a), abs(b)) <= 0.05):
                print(f"{puzzle_index + 1}-{side + 1}_{j + 1}-{k + 1} --> skip")
                continue
            # s_1, s_2 = lead_to_one_size_with_scale(main_side, other_side)

            # Without scale
            s_1, s_2 = lead_to_one_size(main_side, other_side)


            if s_1 is None and s_2 is None:
                print("skip")
                continue

            # M_j = np.array([s_1[0], s_2[1] - s_1[1]])
            # # M_j = np.array([s_2[0], s_1[1] - s_2[1]])
            # m_j_dif = np.array([*[M_j[0][i + 1] - M_j[0][i] for i in range(M_j.shape[1] - 1)], M_j[0][-1] - M_j[0][-2]])
            # M_j = np.vstack((M_j, m_j_dif))
            #
            #
            # # TODO fix quality
            # qulity = 0
            # for ddr in range(len(M_j[1])):
            #     qulity += M_j[1][ddr] ** 2 * M_j[2][ddr]
            # # sum([M_j[1][ind] ** 2 * M_j[2][ind] for ind in range(len(M_j[1]))])

            s_1, s_2, qulity = get_best_quality(s_1, s_2)

            print(f"{puzzle_index + 1}-{side + 1}_{j + 1}-{k + 1} --> {qulity} {main_side[0][-1], other_side[0][-1]}")

            if qulity < min_diff:  # and j == i + 1:
                best_match = [j, k]
                min_diff = qulity


                if j > puzzle_index:
                    display_plot([
                        [s_1, "lines", f"Кривизни {puzzle_index + 1}-{main_index + 1}", "#010BE6", {}, True],
                        [s_2, "lines", f"Кривизни {j + 1}-{k + 1}", "#E6B700", {}, True],
                    ], title=f"{puzzle_index + 1}-{side + 1}_{j + 1}-{k + 1} --> {qulity}",
                        filename=f"curve_comparing/{L}/{puzzle_index + 1}/{puzzle_index + 1}-{side + 1}_{j + 1}-{k + 1}",
                        html=True)

            # if puzzle_index + 1 == j:
            #     display_plot([
            #         [s_1, "lines", f"Кривизни {puzzle_index + 1}-{main_index + 1}", "#010BE6", {}],
            #         [s_2, "lines", f"Кривизни {j + 1}-{k + 1}", "#E6B700", {}],
            #     ], title="Кирвизни")

                # match_data = [np.array(s_1, copy=True), np.array(s_2, copy=True)]

    if [puzzle_index + 1, side + 1] == [1, 3]:
        best_match = [1, 3]

    print(best_match)
    return best_match


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


def lead_to_one_size_with_scale(side_1, side_2):
    min_indexes = find_local_minima(side_1[1]), find_local_minima(side_2[1])
    max_indexes = find_local_maxima(side_1[1]), find_local_maxima(side_2[1])

    if min_indexes[0][1] - min_indexes[0][0] > max_indexes[0][1] - max_indexes[0][0]:
        idx1_1, idx1_2 = min_indexes[0]
        idx2_1, idx2_2 = min_indexes[1]
    else:
        idx1_1, idx1_2 = max_indexes[0]
        idx2_1, idx2_2 = max_indexes[1]


    x1_1, x1_2 = side_1[0, idx1_1], side_1[0, idx1_2]
    x2_1, x2_2 = side_2[0, idx2_1], side_2[0, idx2_2]

    # Вычисляем коэффициенты масштабирования и сдвига:
    # x_new = a * x_old + b
    a = (x2_2 - x2_1) / (x1_2 - x1_1)
    b = x2_1 - a * x1_1

    # Применяем аффинное преобразование
    side_1_scaled = np.copy(side_1)
    side_1_scaled[0] = a * side_1_scaled[0] + b

    # Найдём пересекающийся диапазон X
    min_x = max(np.min(side_1_scaled[0]), np.min(side_2[0]))
    max_x = min(np.max(side_1_scaled[0]), np.max(side_2[0]))
    common_x = np.linspace(min_x, max_x, len(side_1[0]) if len(side_1[0]) > len(side_2[0]) else len(side_2[0]))

    # Интерполяция
    interp_y1 = interp1d(side_1_scaled[0], side_1_scaled[1], kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_y2 = interp1d(side_2[0], side_2[1], kind='linear', bounds_error=False, fill_value='extrapolate')

    aligned_1 = np.vstack([common_x, interp_y1(common_x)])
    aligned_2 = np.vstack([common_x, interp_y2(common_x)])

    return aligned_1[0:, 2500:-2500], aligned_2[0:, 2500:-2500]




display_curve = False


if __name__ == "__main__":
    # puzzles_len = [1, 2, 3]
    puzzles_len = [i for i in range(1, 55)]
    L = "12.0"

    puzzles_reverse, puzzles = load_moments(puzzles_len, L)
    puzzle_sides_data = load_sides(puzzles_len, L)[1]


    if display_curve:
        display_plot([
            [puzzles[display_curve][0], "lines", "Сторона 1", "#E60000", {}, True],
            [puzzles[display_curve][1], "lines", "Сторона 2", "#E6B400", {}, True],
            [puzzles[display_curve][2], "lines", "Сторона 3", "#288E45", {}, True],
            [puzzles[display_curve][3], "lines", "Сторона 4", "#0014E6", {}, True],
        ], title="curvature straight", filename=f"curve_comparing/{L}/0/curvature_straight", html=True)
    
        display_plot([
            [puzzles_reverse[display_curve][0], "lines", "Сторона 1", "#E60000", {}, True],
            [puzzles_reverse[display_curve][1], "lines", "Сторона 2", "#E6B400", {}, True],
            [puzzles_reverse[display_curve][2], "lines", "Сторона 3", "#288E45", {}, True],
            [puzzles_reverse[display_curve][3], "lines", "Сторона 4", "#0014E6", {}, True],
        ], title="curvature reverse", filename=f"curve_comparing/{L}/0/curvature_reverse", html=True)

    current_puzzle_index, shift = find_first_puzzle(puzzles[:2])


    size = (6, 9)
    res_array = [[[] for _ in range(size[1])] for _ in range(size[0])]

    row = 0
    col = 0
    side = 2
    rotate_sign = 1
    con_step_direct = 1

    # current_puzzle_index, shift = [21, 1]

    if current_puzzle_index != 0:
        col = current_puzzle_index // size[1]
        row = current_puzzle_index % size[1]

    res_array[row][col], puzzle_sides_data[current_puzzle_index] = move_sides(deepcopy(puzzles[current_puzzle_index]), puzzle_sides_data[current_puzzle_index], shift)

    for i in range(1, len(puzzles)):
        print(f"\n{i=}, {side=}\n")
        main_side = res_array[row][col][side]

        old_side = side
        col += con_step_direct

        if i % size[1] == 0 and i != 0:
            row += 1
            con_step_direct *= -1
            col += con_step_direct

        if col == size[1] - 1 or col == 0:
            if col == 0 and row % 2 == 1:
                rotate_sign *= -1

            side += rotate_sign
            if side > 3:
                side = 0
            elif side < 0:
                side = 3



        current_puzzle_index, shift = find_next_puzzle(current_puzzle_index, puzzles_reverse, main_side, side)

        new_shift = 4 - ((old_side + 2) % 4 + shift)

        print(f"{current_puzzle_index=}, {shift=}, {side=}, {old_side=}, {new_shift=}")

        res_array[row][col], puzzle_sides_data[current_puzzle_index] = move_sides(
            deepcopy(puzzles[current_puzzle_index]),
            puzzle_sides_data[current_puzzle_index],
            new_shift
        )

        print()



