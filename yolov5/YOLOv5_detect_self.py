import os
import json
import argparse
import cv2
import imghdr
import numpy as np
import torch

# handle numpy data in JSON encoding
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# load the trained model
model = torch.hub.load('.', 'custom', path='../YOLOv5_checkpoint.pt', source='local')

# inference on a single image
def inference(filename):
    file = args.input_dir + filename
    # load the image
    # image = cv2.imread(file)
    # perform inference
    results = model(file)
    # collect the detected objects' information
    boxes = []
    labels = []
    scores = []
    for i in range(len(results.pandas().xyxy[0]['xmin'])):
        #print(results.pandas().xyxy[0])
        x1 = float(results.pandas().xyxy[0]['xmin'][i])
        y1 = float(results.pandas().xyxy[0]['ymin'][i])
        x2 = float(results.pandas().xyxy[0]['xmax'][i])
        y2 = float(results.pandas().xyxy[0]['ymax'][i])
        class_name = results.pandas().xyxy[0]['name'][i]
        class_id = results.pandas().xyxy[0]['class'][i]
        confidence = float(results.pandas().xyxy[0]['confidence'][i])
        if(confidence >= 0.35):
            # boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            boxes.append([x1, y1 ,x2, y2])
            labels.append(int(class_id))
            scores.append(float(confidence))

            # cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,255),1,cv2.LINE_AA)
            # cv2.rectangle(image,(x1,y1-20),(x2,y1),(255,255,255),cv2.FILLED)
            # cv2.putText(image, f"{class_name} {round(confidence, 3)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)


    # add a dummy detection if no objects are detected
    if not boxes:
        boxes = [[-1.0, -1.0, -1.0, -1.0]]
        labels = [-1]
        scores = [-1.0]

    return boxes, labels, scores

# get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../yolov5_datasets/valid/')
parser.add_argument('--output_dir', default='./inference_result/')
args = parser.parse_args()

# perform inference on all images in the input directory
results = {}
for filename in os.listdir(args.input_dir):
    if imghdr.what(os.path.join(args.input_dir, filename)) is not None:  # check is input a image?
        boxes, labels, scores = inference(filename)

        results.update({
            filename: {
                "boxes": boxes,
                "labels": labels,
                "scores": scores
            }
        })

# save the results to a JSON file
with open(os.path.join(args.output_dir, 'pred.json'), 'w') as f:
    json.dump(results, f, cls=NpEncoder, indent=4)