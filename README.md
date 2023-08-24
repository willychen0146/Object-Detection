# **YOLOv5 / DETR object detection tutorial**
## **<font color="red">YOLOv5</font>**
### **<font color="green">Training</font>** (You can refer to [YOLOv5 website](https://github.com/ultralytics/yolov5))
I train YOLOv5 model locally, just follow the steps below (use cmd command line)
***
```cmd
git clone https://github.com/ultralytics/yolov5.git
pip install -r requirements.txt
# you can also use "JSON_to_txt.py" if you need to convert json file into txt file
```
1. Place your datasets under the folder you want to use
2. Modify the path setting in the hw1.yaml under the data folder
3. Go to [YOLOv5 website](https://github.com/ultralytics/yolov5), download the pretrain model (I use the [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt))
```cmd
# run main.py to train the model
python3 train.py --img 640 --epochs 200 --batch-size 12 --data hw1.yaml --weights yolov5x.pt # this is the hyperparameter I use, can be modify yourself
```
***
### **<font color="green">Inference</font>**
You can follow the steps just like hw1.sh
***
### **<font color="green">Draw the bounding boxes on image</font>**
Just using the detect.py (you can modify the path and weight)
```cmd
python3 detect.py --weights ../YOLOv5_checkpoint.pt --source ../yolov5_datasets/test_image
```
***

<div style="break-after: page; page-break-after: always;"></div>

## **<font color="red">DETR</font>**
### **<font color="green">Training</font> (You can refer to [DETR website](https://github.com/facebookresearch/detr))**
I train DETR model locally, just follow the steps below (use cmd command line)
***
```cmd
git clone https://github.com/facebookresearch/detr.git
pip install -r requirements.txt 
```
1. Place your datasets under the folder you want to use, and don't forget to modify the path setting in the main.py
2. Modify the class numbers under the models/detr.py
3. Go to [DETR website](https://github.com/facebookresearch/detr), download the pretrain model (I use the [DETR-R50](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth))
```cmd
# Convert the class numbers in pretrain model (you can modify the class numbers you want in "detr_pretrain_convert_class_to_8.py")
python3 detr_pretrain_convert_class_to_8.py

# run main.py to train the model
python3 main.py --coco_path ../detr_datasets/train/ --epochs 150 --batch_size 2 --resume detr-r50_8.pth # this is the hyperparameter i use, can be modify yourself
```
***
### **<font color="green">Inference</font>**
You can follow the steps just like hw1.sh
***
### **<font color="green">Draw the bounding boxes on image</font>**
Just using the DETR_BBOX_img.py (you can modify the path, weight and class numbers)
```cmd
python3 DETR_BBOX_img.py
```

**The image will look just like this with both method:**
![](.\yolov5\runs\IMG_2468.jpg)