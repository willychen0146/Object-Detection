
'''
detr 單張圖片inference
'''
import cv2
from PIL import Image
import numpy as np
import os
import time
import json
import torch
from torch import nn
# from torchvision.models import resnet50
import torchvision.transforms as T
from hubconf import detr_resnet50, detr_resnet50_panoptic
torch.set_grad_enabled(False)
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 當前使用{}做inference".format(device))

# get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../yolov5_datasets/valid/')
parser.add_argument('--output_dir', default='./inference_result/')
args = parser.parse_args()

# image data processing
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#handle json error
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
# xywh轉xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# 將0-1映射到圖片
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b



# plot box by opencv
def plot_result(pil_img, prob, boxes,save_name=None,imshow=False, imwrite=False):
    LABEL = ["creatures","fish","jellyfish","penguin","puffin","shark","starfish","stingray"]
    len(prob)
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # map class id to class name
    CLASS_MAP = {
        0: "creatures",
        1: "fish",
        2: "jellyfish",
        3: "penguin",
        4: "puffin",
        5: "shark",
        6: "starfish",
        7: "stingray",
    }

    # print(prob)

    # print("-------------------------------")

    # print(boxes)

    if len(prob) == 0:
        print("[INFO] NO box detect !!! ")
        if imwrite:
            if not os.path.exists("./result/pred_no"):
                os.makedirs("./result/pred_no")
            cv2.imwrite(os.path.join("./result/pred_no",save_name),opencvImage)
        return
    cls = []
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):

        cl = p.argmax()
        cls.append(cl)
        label_text = '{}: {}%'.format(LABEL[cl],round(p[cl]*100,2))

        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text,(int(xmin)+10, int(ymin)+30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 0), 2)

    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)

    if imwrite:
        if not os.path.exists("./result/pred"):
            os.makedirs('./result/pred')
        cv2.imwrite('./result/pred/{}'.format(save_name), opencvImage)
    
    return cls




# 單張圖inference
def detect(im, model, transform,prob_threshold=0.7):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    img = img.to(device)
    start = time.time()
    outputs = model(img)
    # keep only predictions with 0.7+ confidence
    # print(outputs['pred_logits'].softmax(-1)[0, :, :-1])
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # print('Pro: ', probas)
    keep = probas.max(-1).values > prob_threshold
    end = time.time()

    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()
    # print('outputs: ',outputs)
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled, end-start



if __name__ == "__main__":
    # detr = DETRdemo(num_classes=3+1)

    detr = detr_resnet50(pretrained=False,num_classes=7+1).eval()  # class+1
    state_dict =  torch.load('../DETR_checkpoint.pth')   # model路徑
    detr.load_state_dict(state_dict["model"])
    detr.to(device)
    

    files = os.listdir(args.input_dir)
    all_list = []
    for file in files:
        print(file)
        if(file[-4:] != '.jpg' and file[-4:] != '.png'):
            continue
        img_path = os.path.join(args.input_dir,file)
        im = Image.open(img_path)

        scores, boxes, waste_time = detect(im, detr, transform)

        # print(scores)
    
        cls = plot_result(im, scores, boxes,save_name=file,imshow=False, imwrite=True)
        #print("[INFO] {} time: {} done!!!".format(file,waste_time))
        print(f'boxes: {boxes}, classes: {cls}, scores: {scores}')
        my_list = []
        for i in range(len(boxes)):
            #if(scores[i]>=0.4):
            #    my_list.append([file,boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3],cls[i],scores[i]])
            my_list.append([file,boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3],cls[i],np.amax(scores[i])])
        all_list.append(my_list)
    

    json_dump = {}

    for i in range(len(all_list)):
        if(len(all_list[i]) == 0):
            print('empty')
            continue
        file_name = all_list[i][0][0]
        class_id = []
        bbox = []
        confidence = []

        for j in range(len(all_list[i])):
            if(all_list[i][j][1] == -1):
                continue 
            bbox.append([all_list[i][j][1],all_list[i][j][2],all_list[i][j][3],all_list[i][j][4]])
            class_id.append(all_list[i][j][5])
            confidence.append(all_list[i][j][6])

        json_dump.update({file_name:{'boxes':bbox,'labels':class_id,'scores':confidence}})

    # with open('pred_detr.json','w') as f:
    #     json.dump(json_dump,f,cls=NpEncoder,indent=2)
    with open(os.path.join(args.output_dir, 'pred_detr.json'), 'w') as f:
        json.dump(json_dump, f, cls=NpEncoder, indent=4)    