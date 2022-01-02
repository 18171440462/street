import linecache
import random
L1 = random.sample(range(1, 3942), 3941)

train = open('/home/ie/桌面/ZDW/projects/traffic_light/dataset/train.txt','w')
test = open('/home/ie/桌面/ZDW/projects/traffic_light/dataset/test.txt','w')
for i in range(0,2470):
    now_line = linecache.getline('/home/ie/桌面/ZDW/projects/traffic_light/dataset/train_dataset.txt', L1[i])
    # now_line = now_line.replace('not_all','after_train')
    print(i,len(now_line),now_line)
    if i < 247:
        test.write(now_line)
    else:
        train.write(now_line)

