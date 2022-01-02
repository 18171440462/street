import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import os, cv2

def lable_id2label_name(id):
    if id == 0:  # 获取单个标注的中文标签
        name = "red"
    elif id == 1:
        name = "yellow"
    elif id == 2:
        name = "green"
    return name

LABELS = ['red', 'yellow', 'green']


def parse_annotation(labels=[]):
    all_imgs = []
    seen_labels = {}
    img = {'object': []}
    txt = open('/home/ie/桌面/ZDW/projects/traffic_light/dataset/train_dataset.txt')
    num = 1
    for eachline in txt:
        num += 1
        print (num)
        each_text = eachline.split()
        image_path = each_text[0]
        image_jpg = image_path.split('/')[-1]
        image_name = image_jpg.split('.')[0]
        image = cv2.imread(image_path)
        img['filename'] = image_name
        img['width'] = int(image.shape[0])
        img['height'] = int(image.shape[1])
        obj = {}
        for i in range(1, len(each_text)):
            each_annotation = each_text[i].split(',')
            x1 = int(each_annotation[0])
            y1 = int(each_annotation[1])
            x2 = int(each_annotation[2])
            y2 = int(each_annotation[3])
            label_id = int(each_annotation[4])
            label_name = lable_id2label_name(label_id)
            # print(label_name)
            obj['name'] = label_name
            if len(labels) > 0 and obj['name'] not in labels:
                break
            else:
                img['object'] += [obj]
            if obj['name'] in seen_labels:
                seen_labels[obj['name']] += 1
            else:
                seen_labels[obj['name']] = 1
            obj['xmin'] = x1
            obj['ymin'] = y1
            obj['xmax'] = x2
            obj['ymax'] = y2
        if len(img['object']) > 0:
            all_imgs += [img]
    return all_imgs, seen_labels

def iou(box, clusters):
    '''
    :param box:      np.array of shape (2,) containing w and h
    :param clusters: np.array of shape (N cluster, 2)
    '''
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def kmeans(boxes, k, dist=np.median, seed=1):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))  ## N row x N cluster
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    # initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # Step 1: allocate each item to the closest cluster centers
        for icluster in range(k):  # I made change to lars76's code here to make the code faster
            distances[:, icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances

## Parse annotations
train_image, seen_train_labels = parse_annotation( labels=LABELS)
# print("N train = {}".format(len(train_image)))
# print(train_image)
# print(seen_train_labels)
y_pos = np.arange(len(seen_train_labels))
fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(1,1,1)
ax.barh(y_pos,list(seen_train_labels.values()))
ax.set_yticks(y_pos)
ax.set_yticklabels(list(seen_train_labels.keys()))
ax.set_title("The total number of objects = {} in {} images".format(
    np.sum(list(seen_train_labels.values())),len(train_image)
))
plt.show()
#
wh = []
for anno in train_image:
    aw = float(anno['width'])  # width of the original image
    ah = float(anno['height']) # height of the original image
    for obj in anno["object"]:
        w = (obj["xmax"] - obj["xmin"])/aw # make the width range between [0,GRID_W)
        h = (obj["ymax"] - obj["ymin"])/ah # make the width range between [0,GRID_H)
        temp = [w,h]
        wh.append(temp)
wh = np.array(wh)
print("clustering feature data is ready. shape = (N object, width and height) =  {}".format(wh.shape))


plt.figure(figsize=(10,10))
plt.scatter(wh[:,0],wh[:,1],alpha=0.3)
plt.title("Clusters",fontsize=20)
plt.xlabel("normalized width",fontsize=20)
plt.ylabel("normalized height",fontsize=20)
plt.show()
