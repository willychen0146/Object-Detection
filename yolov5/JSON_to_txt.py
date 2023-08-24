
# change valid json to txt

import json

path = 'valid/'
filename = '_annotations.coco.json'

with open(path+filename, 'r') as f:
    str0 = f.read()
    #print(str)

strjson = str0
parsedJson = json.loads(strjson)
print(len(parsedJson['images']))

labels_path = 'labels/valid/'
for i in range(len(parsedJson['images'])):
    print('i: ',i)
    
    labels_filename = parsedJson['images'][i]['file_name']
    print(labels_filename[:-4] + '.txt')
    with open(labels_path + labels_filename[:-4] + '.txt', 'w') as f:
        height = parsedJson['images'][i]['height']
        width = parsedJson['images'][i]['width']

        #f.write('height: ' + height + '\n')
        #f.write('width: '  + width  + '\n')

        for j in parsedJson['annotations']:
            if(j['image_id'] == i):
                print(j)
                f.write(str(j['category_id']) + ' ')
                #cx = round((j['bbox'][0]+j['bbox'][2]) / width / 2, 5)
                #cy = round((j['bbox'][1]+j['bbox'][3]) / height / 2, 5)
                cx = round((j['bbox'][0] + j['bbox'][2] / 2 )/ width, 5)
                cy = round((j['bbox'][1] + j['bbox'][3] / 2) / height, 5)
                #ww = round((j['bbox'][2]-j['bbox'][0]) / width, 5)
                #hh = round(abs(j['bbox'][3]-j['bbox'][1]) / height, 5)
                ww = round((j['bbox'][2]) / width, 5)
                hh = round((j['bbox'][3]) / height, 5)
                f.write(str(cx) + ' ')
                f.write(str(cy) + ' ')
                f.write(str(ww) + ' ')
                f.write(str(hh) + '\n')


# change train json to txt       
path = 'train/'
filename = '_annotations.coco.json'

with open(path+filename, 'r') as f:
    str0 = f.read()
    #print(str)

strjson = str0
parsedJson = json.loads(strjson)
print(len(parsedJson['images']))

labels_path = 'labels/train/'
for i in range(len(parsedJson['images'])):
    print('i: ',i)
    
    labels_filename = parsedJson['images'][i]['file_name']
    print(labels_filename[:-4] + '.txt')
    with open(labels_path + labels_filename[:-4] + '.txt', 'w') as f:
        height = parsedJson['images'][i]['height']
        width = parsedJson['images'][i]['width']

        #f.write('height: ' + height + '\n')
        #f.write('width: '  + width  + '\n')

        for j in parsedJson['annotations']:
            if(j['image_id'] == i):
                print(j)
                f.write(str(j['category_id']) + ' ')
                #cx = round((j['bbox'][0]+j['bbox'][2]) / width / 2, 5)
                #cy = round((j['bbox'][1]+j['bbox'][3]) / height / 2, 5)
                cx = round((j['bbox'][0] + j['bbox'][2] / 2 )/ width, 5)
                cy = round((j['bbox'][1] + j['bbox'][3] / 2) / height, 5)
                #ww = round((j['bbox'][2]-j['bbox'][0]) / width, 5)
                #hh = round(abs(j['bbox'][3]-j['bbox'][1]) / height, 5)
                ww = round((j['bbox'][2]) / width, 5)
                hh = round((j['bbox'][3]) / height, 5)
                f.write(str(cx) + ' ')
                f.write(str(cy) + ' ')
                f.write(str(ww) + ' ')
                f.write(str(hh) + '\n')       
        
