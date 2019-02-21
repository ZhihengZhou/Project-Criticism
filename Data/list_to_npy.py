import os
import numpy as np
from PIL import Image
import pickle
import tqdm
import progressbar
import random

os.chdir("./UIdata/Button_List")

test_cases_no = 3000
total_case_counter = 0
max_cases_each_screenshot = 4
max_images_each_app = 10

window_size = 256
window_bound = 240
button_max_size = 128

screenshot_width_min = 0
screenshot_height_min = 33
screenshot_width = 800
screenshot_height = 1216

target_button_type = ["android.widget.Button"]
x = []

apps = os.listdir("./")
apps = [i for i in apps if os.path.isdir(i)]
#apps.reverse()

def get_cases(buttons, d):
    buttons = [i for i in buttons if i[0] in target_button_type]
    buttons = [i for i in buttons if i[1][0] > screenshot_width_min and i[1][1] > screenshot_height_min and i[1][2] < screenshot_width and i[1][3] < screenshot_height]
    buttons = [i[1] for i in buttons]
    buttons = [i for i in buttons if i[2] - i[0] < button_max_size and i[3] - i[1] < button_max_size]
    
    button_group = []
    for b1 in buttons:
        center_window = get_center_window(b1)
        this_group = []
        for b2 in buttons:
            if b2 == b1:
                continue
            b2_x1 = b2[0]
            b2_y1 = b2[1]
            b2_x2 = b2[2]
            b2_y2 = b2[3]
            if b2_x1 > center_window[0] and b2_y1 > center_window[1] and b2_x2 < center_window[2] and b2_y2 < center_window[3]:
                this_group.append(b2)
        button_group.append((b1, len(this_group), this_group, center_window))
    
    button_group.sort(key=takeSecond, reverse=True)
    
    return get_npy_list(button_group[0:max_cases_each_screenshot], d)

def get_center_window(box):
    mid_x = (box[0] + box[2])/2
    mid_y = (box[1] + box[3])/2
    
    x1 = int(mid_x - window_size/2)
    x2 = int(mid_x + window_size/2)
    if x1 < screenshot_width_min:
        x1 = screenshot_width_min
        x2 = x1 + window_size
    elif x2 > screenshot_width:
        x1 = screenshot_width - window_size
        x2 = x1 + window_size
    
    y1 = int(mid_y - window_size/2)
    y2 = int(mid_y + window_size/2)
    if y1 < screenshot_height_min:
        y1 = screenshot_height_min
        y2 = y1 + window_size
    elif y2 > screenshot_height:
        y1 = screenshot_height - window_size
        y2 = y1 + window_size
    
    center_window = (x1,y1,x2,y2)
    return center_window

def get_npy_list(button_group, d):
    cases_npy = []
    for group in button_group:
        
        center_window = group[3]
        target_button = group[0]
        other_buttons = group[2]
        
        target_button_new = [0,0,0,0]
        target_button_new[0] = target_button[0] - center_window[0]
        target_button_new[2] = target_button[2] - center_window[0]
        target_button_new[1] = target_button[1] - center_window[1]
        target_button_new[3] = target_button[3] - center_window[1]
        
        other_buttons_new = []
        for button in other_buttons:
            new_b = [0,0,0,0]
            new_b[0] = button[0] - center_window[0]
            new_b[2] = button[2] - center_window[0]
            
            new_b[1] = button[1] - center_window[1]
            new_b[3] = button[3] - center_window[1]
            other_buttons_new.append(new_b)
        
        cases_npy.append((crop_img(d, center_window), target_button_new, other_buttons_new, center_window))
    return cases_npy

def crop_img(d, window):
    img = Image.open(d + "/origin.png")
    img = img.crop(window)
    img = np.array(img, dtype=np.uint8)
    return img

def takeSecond(elem):
    return elem[1]

### main
pbar = progressbar.ProgressBar()
have_test = False
for app in tqdm.tqdm(apps):
    os.chdir(app)
    dirs = os.listdir()
    dirs = [i for i in dirs if os.path.isdir(i)]
    ramdon.shuffle(dirs)
    dirs = dirs[0:max_images_each_app]
    for d in dirs:
        
        with open(os.path.join(d, "metric.txt"), 'rb') as f:
            buttons = pickle.load(f)
        npy_list = get_cases(buttons, d)
        
        if len(npy_list) > 0:
            x.extend(npy_list)
            total_case_counter += len(npy_list)
    
    os.chdir("../") # To dir "./Data/UIdata/Button_List"

print(total_case_counter, len(x))

ratio = 0.9
p = int(ratio * len(x))
x_train = x[:p]
x_test = x[p:]

max_elements = 20000
if not os.path.exists('../npy-Crop-multi'):
    os.mkdir('../npy-Crop-multi')

# Save test npy
if len(x_test) > max_elements:
    for count in range(int(len(x_test)/max_elements)):
        np.save('../npy-Crop-multi/x_test_' + str(count) + '.npy', x_test[count*max_elements : (count+1)*max_elements])
    np.save('../npy-Crop-multi/x_test_' + str(count+1) + '.npy', x_test[(count+1)*max_elements :])
else:
    np.save('../npy-Crop-multi/x_test.npy', x_test)

# Save load npy
if len(x_train) > max_elements:
    for count in range(int(len(x_train)/max_elements)):
        np.save('../npy-Crop-multi/x_train_' + str(count) + '.npy', x_train[count*max_elements : (count+1)*max_elements])
    np.save('../npy-Crop-multi/x_train_' + str(count+1) + '.npy', x_train[(count+1)*max_elements :])
else:
    np.save('../npy-Crop-multi/x_train.npy', x_train)

print("Process finished!!!")






