import json
import numpy as np
import requests
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# for l in [5, 8, 10, 12, 15]:
for l in [5.5]:
# for l in [8]:
#     for i in [28, 33, 40, 46]:
    for i in range(1, 55):
    # for i in [5]:
    # for i in [3, 20]:
    #     for direct in [0, 1]:
        for direct in [0]:
            filename = f"all_good_{i}.txt"
            print(filename)

            arr = np.loadtxt(f"../PDF/res_data/{filename}")
            arr_expanded = np.hstack((arr, np.zeros((arr.shape[0], 1))))

            data = {
                "data": json.dumps(arr_expanded.tolist()),
                "iter_count": 4,
                "start_value": 0.01,
                "end_value": 0.0005,
                "curve_type": "loop",
                "save_data": 0, # TODO change
                "file_name": filename,
                "straight": direct,
                "puzzle_index": i,
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
