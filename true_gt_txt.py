import os


def lable_id2label_name(id):
    if id == 0:  # 获取单个标注的中文标签
        name = "red"
    elif id == 1:
        name = "yellow"
    elif id == 2:
        name = "green"
    return name

if not os.path.exists("./map_out/ground-truth"):
    os.makedirs("./map_out/ground-truth")

txt = open('/home/ie/桌面/ZDW/projects/traffic_light/dataset/test.txt')
num = 0
for eachline in txt:
    num +=1
    print(num)
    each_text = eachline.split()
    image_path = each_text[0]
    image_jpg = image_path.split('/')[-1]
    image_name = image_jpg.split('.')[0]
    f = open("./map_out/ground-truth/" + image_name + ".txt", "w")
    for i in range(1, len(each_text)):
        each_annotation = each_text[i].split(',')
        x1 = int(each_annotation[0])
        y1 = int(each_annotation[1])
        x2 = int(each_annotation[2])
        y2 = int(each_annotation[3])
        label_id = int(each_annotation[4])
        label_name = lable_id2label_name(label_id)
        f.write("%s %s %s %s %s \n" % (label_name, x1, y1, x2, y2))