import json
import numpy as np
import requests
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from find_corner_points import fix_all_between_corners, split_contour_by_corners

all_corners = {}
with open("top_4_candidates.txt") as f:
    corners = f.read().splitlines()

    for indd, co in enumerate(corners, 1):
        all_corners[indd] = list(map(int, co.split(" ")))

# for l in [5, 8, 10, 12, 15]:
for l in [5, 8, 10]:
# for l in [15.0999]:
# for l in [10.1]:
# for l in [10.0999]:
# for l in [5]:
    for i in range(1, 55):
    # for i in [44]:
#     for i in [5, 8, 9, 44]:
#     for i in [8, 9, 44]:
    # for i in [5]:
        for direct in [0, 1]:
        # for direct in [0]:
            filename = f"all_good_{i}.txt"
            print(filename)

            arr = np.loadtxt(f"../PDF/res_data/{filename}").T

            contour = np.stack(arr, axis=1).astype(np.int32).reshape(-1, 1, 2)[::-1]
            corners = all_corners[i]

            fixed_contour = fix_all_between_corners(contour, corner_indices=corners, offset=25, window=50)

            input_arrays = split_contour_by_corners(fixed_contour, corner_indices=corners)


            for side, arr_side in enumerate(input_arrays):
                arr_expanded = np.hstack((arr_side, np.zeros((arr_side.shape[0], 1))))
                data = {
                    "data": json.dumps(arr_expanded.tolist()),
                    "iter_count": 9,
                    "start_value": 0.01,
                    "end_value": 0.0005,
                    "curve_type": "not_loop",
                    # "curve_type": "loop",
                    "save_data": 1, # TODO change
                    "file_name": filename,
                    "straight": direct,
                    "puzzle_index": i,
                    "general_l": l,
                    "side": side,
                }

                resp = requests.post("http://127.0.0.1:5254/calculate", data=data)


                for i in range(len(resp.json()["plots"])):
                    img_data = base64.b64decode(resp.json()["plots"][i])
                    img = Image.open(BytesIO(img_data))

                    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                    ax.imshow(img)
                    ax.axis("off")

                    plt.show()
