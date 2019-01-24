import os
from PIL import Image, ImageDraw, ImageFile
import numpy as np
import pickle

os.chdir("UIdata/Button_List")

apps = os.listdir()
apps = [i for i in apps if os.path.isdir(i)]
print(len(apps))

imgs_no = []
buttons_no = []

for app in apps:
    dirs = os.listdir(app)
    dirs = [i for i in dirs if os.path.isdir(i)]
    imgs_no.append(len(dirs))
    for d in dirs:
        with open(os.path.join(app, d, "metric.txt"), 'rb') as f:
            metric = pickle.load(f)
            buttons_no.append(len(metric))
            
os.chdir("../../")
np.save("imgs_info.npy", imgs_no)
np.save("buttons_info.npy", buttons_no)
