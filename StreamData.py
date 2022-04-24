import torch
from torch.utils.data.dataset import Dataset
import json
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2

def get_labels():
    #train_json = json.load(open(r'E:\downloads\Argoverse-HD\annotations\train.json','r',encoding='utf-8'))
    train_json = json.load(open('/hy-tmp/Argoverse-HD/annotations/train.json', 'r', encoding='utf-8'))
    annolists =[[] for _ in range(39384)]
    for j in train_json['annotations']:
        annolists[j['image_id']].append(j)
    return annolists


def make_triplet_sets():
    #train_json = json.load(open(r'E:\downloads\Argoverse-HD\annotations\train.json','r',encoding='utf-8'))
    train_json = json.load(open('/hy-tmp/Argoverse-HD/annotations/train.json', 'r', encoding='utf-8'))
    img_jsons = train_json['images']
    img_json_lists = [[] for _ in range(65)]
    for img_json in img_jsons:
        sid = int(img_json['sid'])
        img_json_lists[sid].append(img_json)
    triplet_lists = []
    for i in range(65):
        triplet = {'t-1':[],'t':[],'t+1':[]}
        img_json_list = img_json_lists[i]
        triplet['t-1'] = img_json_list[:-2]
        triplet['t'] = img_json_list[1:-1]
        triplet['t+1'] = img_json_list[2:]
        triplet_lists.append(triplet)
    triplet_sets = []
    for triplet_dict in triplet_lists:
        for j in range(len(triplet_dict['t-1'])):
            last = triplet_dict['t-1'][j]
            now = triplet_dict['t'][j]
            future = triplet_dict['t+1'][j]
            triplet_sets.append([last,now,future])
    return triplet_sets

class StreamDataSet(Dataset):
    def __init__(self,triplet_sets):
        self.triplet_sets = triplet_sets
        self.annolists = get_labels()
        self.trans = transforms.Compose([
            #transforms.Resize((600, 960)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        last,now,future = self.triplet_sets[item]
        sid = last['sid']
        #lastimg = Image.open('E:\\downloads\\Argoverse-1.1\\tracking\\train\\'+str(sid)+'\\ring_front_center\\'+last['name'])
        lastimg = Image.open('/hy-tmp/Argoverse-1.1/tracking/train/' + str(sid) + '/ring_front_center/' + last['name'])
        #nowimg = Image.open('E:\\downloads\\Argoverse-1.1\\tracking\\train\\'+str(sid)+'\\ring_front_center\\'+now['name'])
        nowimg = Image.open('/hy-tmp/Argoverse-1.1/tracking/train/' + str(sid) + '/ring_front_center/' + now['name'])
        lastimg = lastimg.resize((960, 600), Image.BICUBIC)
        mask_last = Image.new(mode='RGB', size=(960, 960), color=(128, 128, 128))
        mask_last.paste(lastimg, (0, 0))
        nowimg = nowimg.resize((960, 600), Image.BICUBIC)
        mask_now = Image.new(mode='RGB', size=(960, 960), color=(128, 128, 128))
        mask_now.paste(nowimg, (0, 0))
        lastimg = self.trans(mask_last)
        nowimg = self.trans(mask_now)
        #cv2.imwrite('show.jpg', (nowimg * 255).numpy().transpose(1, 2, 0)[..., ::-1])
        nowlabels = self.annolists[int(now['id'])]
        futurelabels = self.annolists[int(future['id'])]
        now_boxes,future_boxes = [],[]
        for label_json_now in nowlabels:
            x1,y1,w,h = label_json_now['bbox']
            x2 = x1 + w
            y2 = y1 + h
            x1 = float(x1) /2
            y1 = float(y1) /2
            x2 = float(x2) /2
            y2 = float(y2) /2
            cx = (x1 + x2) /2
            cy = (y1 + y2) /2
            cw = x2 - x1
            ch = y2 - y1
            cls = int(label_json_now['category_id'])
            label_now = np.array([cx,cy,cw,ch,cls],dtype=np.float)
            now_boxes.append(label_now)
        for label_json_future in futurelabels:
            x1,y1,w,h = label_json_future['bbox']
            x2 = x1 + w
            y2 = y1 + h
            x1 = float(x1) /2
            y1 = float(y1) /2
            x2 = float(x2) /2
            y2 = float(y2) /2
            cx = (x1 + x2) /2
            cy = (y1 + y2) /2
            cw = x2 - x1
            ch = y2 - y1
            cls = int(label_json_future['category_id'])
            label_future = np.array([cx,cy,cw,ch,cls],dtype=np.float)
            future_boxes.append(label_future)
        now_boxes = np.array(now_boxes)
        future_boxes = np.array(future_boxes)
        return lastimg,nowimg,torch.Tensor(now_boxes),torch.Tensor(future_boxes)

    def __len__(self):
        return len(self.triplet_sets)

if __name__ == '__main__':
    triplet_sets = make_triplet_sets()
    dataset = StreamDataSet(triplet_sets)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    print(dataset[0][3].shape)
