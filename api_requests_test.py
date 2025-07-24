import json
import numpy as np
import requests
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from find_corner_points import fix_all_between_corners, split_contour_by_corners


arr = np.loadtxt(f"test_points.txt")
l = 1.1

arr_expanded = np.hstack((arr, np.zeros((arr.shape[0], 1))))
data = {
    "data": json.dumps(arr_expanded.tolist()),
    "iter_count": 9,
    "start_value": 0.01,
    "end_value": 0.0005,
    "curve_type": "not_loop",
    # "curve_type": "loop",
    "save_data": 1, # TODO change
    "file_name": "test_points.txt",
    "straight": 1,
    "puzzle_index": 0,
    "general_l": l,
}

resp = requests.post("http://127.0.0.1:5254/calculate", data=data)


for i in range(len(resp.json()["plots"])):
    img_data = base64.b64decode(resp.json()["plots"][i])
    img = Image.open(BytesIO(img_data))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.imshow(img)
    ax.axis("off")

    plt.show()
