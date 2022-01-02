
from PIL import Image
from tqdm import tqdm
from yolo import YOLO
import json
import numpy as np
import os

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

yolo = YOLO()
result = {}
total = []
save_json_path = "./result.json"
dir = "/home/ie/桌面/ZDW/projects/person_car/documents/test_images/"
image_ids = os.listdir(dir)
for image_path in tqdm(image_ids):
    filename = "test_images\\"+ image_path
    image = Image.open(dir+image_path)
    objs = yolo.get_json(image)
    for obj in objs: #predicted_class,score,left, top, right, bottom
        each_obj = {}
        each_obj["filename"] = filename
        each_obj["box"] = {}
        each_obj["box"]["xmin"] = obj[2]
        each_obj["box"]["ymin"] = obj[3]
        each_obj["box"]["xmax"] = obj[4]
        each_obj["box"]["ymax"] = obj[5]
        each_obj["label"] = obj[0]
        each_obj["conf"] = obj[1]
        total.append(each_obj)
print(total)
result["annotations"] = total
json.dump(result, open(save_json_path, 'w'), indent=4, cls=MyEncoder)