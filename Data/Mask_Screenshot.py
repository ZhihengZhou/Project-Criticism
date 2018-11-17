import os
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET 
import re
import hashlib
from skimage import transform
import progressbar
import difflib
import pickle
    
def mask_and_save_image(box):
    """
    This function take a mask range and mask the range with wite block and save image.
    
    Parameter:
    box : A four element list which note a rectangle range of image. 
          Example: (upper-left x, upper-left y, botton-right x, botton-y)
    """
    
    global count
    global Button_list
    
    width = box[2]-box[0]
    height = box[3]-box[1]
    Button_list.append((width, height))
    
    if width >= 0 and width <= local_window and height >= 0 and width <= local_window:
    
        try:
            im = Image.open(dir_name + "/" + img_name)
            img_names = img_name.split('.')
            region=im.crop(box)
            
            # Crop screenshot to global window
            mid_x = (box[0] + box[2])/2
            mid_y = (box[1] + box[3])/2
        
            x1 = int(mid_x - global_window/2)
            if x1 < image_width_min:
                x1 = image_width_min
            elif x1 > image_width - global_window:
                x1 = image_width - global_window
            
            box[0] = box[0] - x1
            box[2] = box[2] - x1
        
            y1 = int(mid_y - global_window/2)
            if y1 < image_height_min:
                y1 = image_height_min
            elif y1 > image_height - global_window:
                y1 = image_height - global_window
            
            box[1] = box[1] - y1
            box[3] = box[3] - y1
            
            if box[1] < 0:
                return
            
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            
            im = im.crop((x1, y1, x1 + global_window, y1 + global_window))
            
            mask = Image.new("RGBA",(width,height),(127,127,127))
            im.save(save_dir +  img_names[0] + "-mask_" + str(count) + "-" + str(box) + "." + img_names[1])
            
            im.paste(mask,box)
            #im.save(save_dir +  img_names[0] + "-origin_" + str(count) + "-" + str(box) + "." + img_names[1])
            
            count += 1
        except Exception:
            #print(save_dir + img_names)
            pass
    
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
    if attribute in node.attrib and node.attrib[attribute] == find:
        # Operations after find the target XML objects.
        bounds = node.attrib['bounds']
        bounds = re.findall(r'(\w*[0-9]+)\w*',bounds)
        bounds = [int(i) for i in bounds]
        mask_and_save_image(bounds)
    
    # Visit all the target objects in current object.
    for n in node.findall(element_name):
        find_all_element_by_attribute(n, element_name, attribute, find)

def get_file_md5(f):
    m = hashlib.md5()
    while True:
        #如果不用二进制打开文件，则需要先编码
        data = f.readline()
        data = delete_attributes_by_name(data,["class"]).encode('utf-8')
        # data = f.read(1024)  #将文件分块读取
        if not data:
            break
        m.update(data)
        # print(data)
    return m.hexdigest()

def delete_attributes_by_name(object_string, keep_list):
    """
    This function takes an XML objects string (a line) and the attributes name which user want to keep. Then return 
    the cleaned XML objects string with only the attributs in keep list. And keep the XML object hierarchy.
    
    Parameters:
    object_string : And XML object string (a line in xml file).
    keep_list : An list of attributes name which want to keep.
    
    Return:
        The cleaned XML object string.
    """
    
    object_string = object_string.replace("*", "#");
    
    attributes = re.findall(r' [^<|^"|^\']*?=["|\']', object_string)
    # print(attributes)
    attributes = [re.findall(r'[^ |^=]+', x) for x in attributes]
    # print(attributes)
    attributes = [x for x in attributes if x[0] not in keep_list]
    
    for attr in attributes:
        try:
            sub_attr = " " + attr[0] + "=" + attr[1] + ".*?" + attr[1]
            object_string = re.sub(sub_attr, "", object_string)
        except Exception:
            pass

    # print(object_string)
    return object_string

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
os.chdir("./UIdata") 
filenames = os.listdir("./")
dirs = [d for d in filenames if "-output" in d]

output_dir = "./Masked-Crop/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
# Progressbar
Max = 100000
# pbar = progressbar.ProgressBar(maxval = Max).start()
pbar = progressbar.ProgressBar()

# Parameters
break_flag = False
image_count = 0
masked_count = 0
local_window = 64
global_window = 256
image_width_min = 0
image_height_min = 33
image_width = 800
image_height = 1216

visited_screenshot = []
visited_screenshot_md5 = []

last_app_domain = None
Button_list = []

for app_dir in pbar(dirs):
    
    if break_flag:
        break
    
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
        print(current_app_domain, last_app_domain)
    last_app_domain = current_app_domain
    
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
            
            ### MD5 version
            # Check duplicate screenshot
#             with open(dir_name + "/" + xml_name[0]) as f:
#                 file_md5 = get_file_md5(f)
#                 if file_md5 in visited_screenshot_md5:
#                     continue
#                 else:
#                     visited_screenshot.append((file_md5, app_dir, i))
#                     visited_screenshot_md5.append(file_md5)

              ### String similarity version
            visited_flag = False
            # Check duplicate screenshot
            with open(dir_name + "/" + xml_name[0]) as f:
                class_str = get_attributes(f)
                for visited in visited_screenshot:
                    seq = difflib.SequenceMatcher(None, class_str, visited[0])
                    ratio = seq.ratio()
                    if ratio > 0.95:
                        
                        visited_flag = True
                        break

            if visited_flag:
                continue
            
            visited_screenshot.append((class_str, app_dir, i))
            
            # Start to mask screenshot
            count = 0
            img_name = i
            find_all_element_by_attribute(root, "node", "class", "android.widget.Button")
            
            # Save original image
            if count > 0:
                im = Image.open(dir_name + "/" + img_name)
                im.save(save_dir + img_name)
                image_count += 1
                masked_count += count
                
#             if image_count >= Max:
#                 pbar.update(Max)
#                 break_flag = True
#                 break
                
            # pbar.update(masked_count)
            
with open("./button_size.txt", "wb") as fp:   #Pickling
    pickle.dump(Button_list, fp)
    
print(len(Button_list))
            
print("Have masked " + str(image_count) + " screenshots, gets " + str(masked_count) + " masked screenshots.")
