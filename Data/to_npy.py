import glob
import os
import cv2
import numpy as np
import re
from PIL import Image
import progressbar


ratio = 0.95

max_elements = 20000

x = []

os.chdir("./UIdata/Masked-Crop/") 
paths = os.listdir('./')
paths = [d for d in paths if not os.path.isfile(d)]

pbar = progressbar.ProgressBar()

for path in pbar(paths):
    
    imgs = os.listdir(path)
    imgs_masked = [i for i in imgs if "mask" in i]
    for img_masked in imgs_masked:
        
        img_id = img_masked.split("-")[0]
        
        mask_bounds = img_masked.split("-")[-1].split(".")[0]
        mask_bounds = re.findall(r'(\w*[0-9]+)\w*',mask_bounds)
        mask_bounds = [int(i) for i in mask_bounds]

        mask_bounds = np.array(mask_bounds, dtype=np.uint8)

        img1 = Image.open(os.path.join(path, img_masked))
        img1 = np.array(img1, dtype=np.uint8)

        x.append((img1, mask_bounds))


x = np.array(x)
np.random.shuffle(x)

p = int(ratio * len(x))
x_train = x[:p]
x_test = x[p:]

os.chdir("../") 

if not os.path.exists('./npy-Crop'):
    os.mkdir('./npy-Crop')
    
if len(x_test) > max_elements:
    for count in range(int(len(x_test)/max_elements)):
        np.save('./npy-Crop/x_test_' + str(count) + '.npy', x_test[count*max_elements : (count+1)*max_elements])
    np.save('./npy-Crop/x_test_' + str(count+1) + '.npy', x_train[(count+1)*max_elements :])
else:
    np.save('./npy-Crop/x_test.npy', x_test)

if len(x_train) > max_elements:
    for count in range(int(len(x_train)/max_elements)):
        np.save('./npy-Crop/x_train_' + str(count) + '.npy', x_train[count*max_elements : (count+1)*max_elements])
    np.save('./npy-Crop/x_train_' + str(count+1) + '.npy', x_train[(count+1)*max_elements :])
else:
    np.save('./npy-Crop/x_train.npy', x_train)

pbar.finish()

print("Done.")
