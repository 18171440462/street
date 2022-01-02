import json

path = "./documents/train.json"
dir = "/home/ie/桌面/ZDW/projects/person_car/documents/train_images/"
file_write_obj = open("./dataset/dataset.txt", 'w')  # 新文件
last_name = 'oriname'
num = 0
classes = ["van","Truck","Pickup","Car","MediumBus","Pedestrian","BiCyclist","TricycleClosed","OtherCar","PersonSitting","LightTruck","TricycleOpenHuman","HeavyTruck","MMcar","EngineTruck","LargeBus","TricycleOpenMotor","Bike","CampusBus","Machineshop","MotorCyclist","Motorcycle"]
with open(path,"r") as f:
    files = json.load(f)
    annotations = files["annotations"]
    for each_image in annotations:
        label_name = dir + each_image["filename"].split("\\")[-1]
        label_class = each_image["label"]
        label_class = classes.index(label_class)
        box = each_image["box"]
        x1 = str(int(box["xmin"]))
        y1 = str(int(box["ymin"]))
        x2 = str(int(box["xmax"]))
        y2 = str(int(box["ymax"]))
        print(label_name,label_class)
        each_object = " "+x1+","+y1+","+x2+","+y2+","+str(label_class)
        if label_name != last_name:
            file_write_obj.write('\n')
            file_write_obj.write(str(label_name)+each_object)
        else:
            file_write_obj.write(each_object)
        last_name = label_name