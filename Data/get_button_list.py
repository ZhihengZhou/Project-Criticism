import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET 
import re
import hashlib
from skimage import transform
import pickle
import progressbar
import difflib
# import imagehash
import distance
import tqdm

# Find node recursively
def find_all_element_by_attribute(node, element_name, attribute, find):
    """
    This function use preorder traversal to find the all the XML objects by its attributs value.
    
    Parameters:
    node        : XML object.
    element_name: Target object name (name of node).
    attribute   : Target attribut name in XML object.
    find        : Target attribute value which mean to find.
    """
    global bound_list
    global count
    
    if attribute in node.attrib and find in node.attrib[attribute]:
        # Operations after find the target XML objects.
        bounds = node.attrib['bounds']
        bounds = re.findall(r'(\w*[0-9]+)\w*',bounds)
        bounds = [int(i) for i in bounds]
        
        
        box = bounds
        width = box[2]-box[0]
        height = box[3]-box[1]
        if width > 0 and height > 0:
            count += 1
            bound_list.append((node.attrib[attribute], bounds))
        
        # mask_and_save_image(bounds)
    
    # Visit all the target objects in current object.
    for n in node.findall(element_name):
        find_all_element_by_attribute(n, element_name, attribute, find)
        
def get_dhash(img):
    return str(imagehash.dhash(img))

def get_attributes(f):
    
    class_list = []
    while True:
        data = f.readline()
        
        class_type = re.findall(r'class=".*?"', data)
        if len(class_type) > 0:
            class_type = re.sub("class=", "", class_type[0])
            class_type = re.sub('\"', "", class_type)
            class_type = re.sub("android.", "", class_type)
            class_list.append(class_type)
        if not data:
            break
    return ' '.join(class_list)

# Main
os.chdir("UIdata") 
filenames = os.listdir("./")
dirs = [d for d in filenames if "-output" in d]

output_dir = "./Button_List/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
pbar = progressbar.ProgressBar()

last_app_domain = ""
last_app_name = ""
image_count = 0
masked_count = 0

for app_dir in tqdm.tqdm(dirs):
    
    app_name = app_dir.split("-")[0]
    dir_name = os.path.join(app_dir, "stoat_fsm_output", "ui")
    save_dir = output_dir + app_name + "/"
    
    # Check the domain of current and last app.
    # Only do the remove dupilicate process for the same domain app.
    current_app_domain = app_name.split(".")[0:2]
    if current_app_domain != last_app_domain:
        visited_screenshot = []
        visited_screenshot_md5 = []
    else:
        print(app_name, last_app_name)
    last_app_domain = current_app_domain
    last_app_name = app_name
    
    files_names = os.listdir(dir_name)
    imgs = [d for d in files_names if "png" in d]
    
    for i in imgs:
        
        xml_name = [d for d in files_names if d == i.split(".")[0] + ".xml"]
        if len(xml_name) > 0:
            
            tree = ET.parse(dir_name + "/" + xml_name[0]) 
            root = tree.getroot()
            
            # Mask specific screen diretion image
            # '0' for only vertical, '1' for only horizon, '2' for both
            if root.attrib['rotation'] != '0':
                continue
            
            ### Remove similar ui
            ## image dhash
#             im = Image.open(dir_name + "/" + i)
#             dhash_value = get_dhash(im)
#             if dhash_value in visited_screenshot:
#                 continue
#             visited_screenshot.append(dhash_value)
            
           
            ## String similarity version
            visited_flag = False
            # Check duplicate screenshot
            with open(dir_name + "/" + xml_name[0]) as f:
                class_str = get_attributes(f)
                for visited in visited_screenshot:
                    seq = difflib.SequenceMatcher(None, class_str, visited[0])
                    ratio = seq.ratio()
                    if ratio > 0.90:
                        visited_flag = True
                        break
                        
            if visited_flag:
                continue
            visited_screenshot.append((class_str, app_dir, i))
            
            ### mask screenshot
            count = 0
            img_name = i
            bound_list = []
            find_all_element_by_attribute(root, "node", "class", "Button")
            
            # Save original image
            if count > 0 and len(bound_list) == count:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                
                img_names = img_name.split('.')
                save_dir_current = save_dir + img_names[0] + "/"
                if not os.path.exists(save_dir_current):
                    os.mkdir(save_dir_current)
                
                # Save original image
                im = Image.open(dir_name + "/" + img_name)
                im.save(save_dir + img_name)
                im.save(save_dir_current + "origin." + img_names[1])
                image_count += 1
                masked_count += count
                
                # Save button list
                with open(save_dir_current + "metric.txt", "w") as fp:
                    fp.write(str(bound_list))
                    # pickle.dump(bound_list, fp, protocol=2)
                
                # Draw bounds on image
                drawObject = ImageDraw.Draw(im)
                for b in bound_list:
                    drawObject.rectangle(b[1], fill=(255,0,0))
                im.save(save_dir_current + "buttons_on." + img_names[1])
                    

print("Have masked " + str(image_count) + " screenshots, gets " + str(masked_count) + " masked screenshots.")





