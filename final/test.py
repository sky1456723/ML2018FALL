import numpy as np
import pandas as pd
import math
import os
from PIL import Image
import torch
import torch.utils
import torch.nn.functional as F
import torchvision
import gc
import sys

### DEVICE ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = os.path.join("../","data","ntu_final_data")

test_file = pd.read_csv(os.path.join(root_dir,"medical_images","test.csv"))
test_data = []
for i in test_file.index:
    test_data.append(test_file.loc[i]["Image Index"])
img_dirs = os.path.join(root_dir,"medical_images","images")
                      
def get_dataloader(data_list):
    part_data = []
    print(len(data_list))
    for i, name in enumerate(data_list):
        print(i,end='\r')
        img = Image.open(os.path.join(img_dirs, name))
        img = img.convert(mode="L")
        img = img.resize((512,512))
        img = np.array(img)
        img = np.expand_dims(img, axis=0) / 255
        part_data.append(img)

    batch = 4
    label_data_x = torch.Tensor(part_data)
    label_dataset = torch.utils.data.TensorDataset(label_data_x)
    label_dataloader = torch.utils.data.DataLoader(dataset = label_dataset,
                                                   batch_size =batch,
                                                   shuffle = False,
                                                   num_workers = 1 )
    del part_data
    return label_dataloader

def get_dataloader_RGB(data_list, transform=None, normalize=None):
    part_data = []
    part_label = []
    print(len(data_list))
    for i, pair in enumerate(data_list):
        print(i,end='\r')
        img = Image.open(os.path.join(img_dirs, pair[0]))
        if transform != None:
            img = img.convert(mode="RGB")
            img = transform(img)
            img = np.array(img)/255
            img = np.transpose(img, axes=[2,0,1])
        else:
            img = img.convert(mode="RGB")
            img = img.resize((224,224))
            img = np.array(img) / 255
            img = np.transpose(img, axes=[2,0,1])
        label = pair[1].split()
        label = np.array([int(c) for c in label])
        part_label.append(label)
        part_data.append(img)

    batch = 4
    label_data_x = torch.Tensor(part_data)
    label_data_y = torch.Tensor(part_label)
    label_dataset = torch.utils.data.TensorDataset(label_data_x, label_data_y)
    label_dataloader = torch.utils.data.DataLoader(dataset = label_dataset,
                                                   batch_size =batch,
                                                   shuffle = False,
                                                   num_workers = 1 )
    del part_data, part_label
    return label_dataloader

input_size = sys.argv[3]
model_name = sys.argv[1]
model_path = os.path.join("./",model_name)
model = torch.load(model_path).to(device)
model = model.eval()

count = 0
last = False
ans_list = []
class_name = open(os.path.join(root_dir,"medical_images","classname.txt")).readlines()
first_row=["id"]
first_row.extend([name.replace("\n","") for name in class_name])
#ans_list.append(first_row)
while not last:
    if len(test_data[count*1000:(count+1)*1000]) == 0:
        break
    print("Part ",count)
    if input_size == 224:
        dataloader = get_dataloader_RGB(test_data[count*1000:(count+1)*1000])
    else:
        dataloader = get_dataloader(test_data[count*1000:(count+1)*1000])
    count += 1
    for i, data in enumerate(dataloader):
        print("Batch: ", i, end='\r')
        pred = model.output_act( model(model.pre_conv(data[0].to(device))) )
        for one_row in pred.cpu().data.numpy():
            ans_list.append(one_row)
            
output_id = np.expand_dims(np.array(test_data),axis=1)
output = np.hstack((output_id, ans_list))
output = np.vstack((first_row, output))
output = pd.DataFrame(output)
output.to_csv(sys.argv[2], index=None, header=None)