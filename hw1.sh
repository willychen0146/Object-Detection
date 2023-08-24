input=$1 # testing images directory with images named 'xxxx.jpg (e.g. input/test_dir/)
output=$2 # path of output json file (e.g. output/pred.json)

# inference (YOLOv5): inference locally and output pred.json
python3 YOLOv5_detect_self.py --input_dir $1 --output_dir $2

# inference (DETR): inference locally and output pred.json
# cd detr
# python3 DETR_BBOX_img.py --input_dir $1 --output_dir $2
# cd..
