import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
import random
from PIL import Image
sys.path.append('..')
from network256 import Network
from load import load_test, load


# Hyperparameters
IMAGE_SIZE = 256
LOCAL_SIZE = 64
BATCH_SIZE = 16

test_data = load_test()
print(len(test_data))
test_data = [x for x in test_data if len(x[1]) == 4]
print(len(test_data))
test_data = [x for x in test_data if (int(x[1][2]) - int(x[1][0]) > 0 and int(x[1][3]) - int(x[1][1]) > 0)]
print(len(test_data))

# Load train and test data.
train_data, test_data = load("../../Data/UIdata/npy-Crop/")

print(len(train_data))
train_data = [x for x in train_data if len(x[1]) == 4]
print(len(train_data))

train_data = [x for x in train_data if (int(x[1][2]) - int(x[1][0]) > 0 and int(x[1][3]) - int(x[1][1]) > 0)]
print(len(train_data))

print(len(test_data))
test_data = [x for x in test_data if len(x[1]) == 4]
if len(test_data) < BATCH_SIZE:
    test_data = train_data
print(len(test_data))

test_data = [x for x in test_data if (int(x[1][2]) - int(x[1][0]) > 0 and int(x[1][3]) - int(x[1][1]) > 0)]
print(len(test_data))

def test():
    
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    x_modified = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x, x_modified, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()
    saver.restore(sess, '../backup/latest')

    np.random.shuffle(test_data)
    

    step_num = int(len(test_data) / BATCH_SIZE)

    cnt = 0
    for i in tqdm.tqdm(range(step_num)):
        test_batch = test_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        
        x_batch = np.array([i[0] for i in test_batch])
        x_batch = np.array([a / 127.5 - 1 for a in x_batch])
        
        x_batch_modified = modify_images(test_batch)
        x_batch_modified = np.array([a / 127.5 - 1 for a in x_batch_modified])
        
        bounds = np.array([i[1] for i in test_batch])
        _, mask_batch = get_points(bounds)
        completion = sess.run(model.imitation, feed_dict={x: x_batch, x_modified: x_batch_modified, mask: mask_batch, is_training: False})
        for i in range(BATCH_SIZE):
            cnt += 1
            raw = x_batch[i]
            raw = np.array((raw + 1) * 127.5, dtype=np.uint8)
            
            # masked = raw * (1 - mask_batch[i]) + np.ones_like(raw) * mask_batch[i] * 255
            modified = x_batch_modified[i]
            modified = np.array((modified + 1) * 127.5, dtype=np.uint8)
            
            img = completion[i]
            img = np.array((img + 1) * 127.5, dtype=np.uint8)
            
            dst = './output/{}.jpg'.format("{0:06d}".format(cnt))
            output_image([['Input', modified], ['Output', img], ['Ground Truth', raw]], dst, bounds[i])


def get_points(bounds):
    points = []
    mask = []
    for b in bounds:
        
        b = [int(x) for x in b]
        mid_y = (b[0] + b[2])/2
        mid_x = (b[1] + b[3])/2
        
        x1 = int(mid_x - LOCAL_SIZE/2)
        if x1 < 0:
            x1 = 0
        elif x1 > IMAGE_SIZE - LOCAL_SIZE:
            x1 = IMAGE_SIZE - LOCAL_SIZE
        
        y1 = int(mid_y - LOCAL_SIZE/2)
        if y1 < 0:
            y1 = 0
        elif y1 > IMAGE_SIZE - LOCAL_SIZE:
            y1 = IMAGE_SIZE - LOCAL_SIZE
    
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])
        
        p1 = b[0]
        q1 = b[1]
        p2 = b[2]
        q2 = b[3]
        
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)
    
    
    return np.array(points), np.array(mask)
    

def output_image(images, dst, box):
    fig = plt.figure(figsize=(12,4))
    for i, image in enumerate(images):
        text, img = image
        fig.add_subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.gca().add_patch(plt.Rectangle((box[0],box[1]),box[2] - box[0],box[3] - box[1],linewidth=1,edgecolor='g',facecolor='none'))
        plt.xlabel(text)
    plt.savefig(dst)
    plt.close()
    
def change_color(img, box):
    button = img[box[1]:box[3]+1,box[0]:box[2]+1]
    shift = random.randint(-50, 50)
    channel = random.randint(0, 2)
    button[:,:,channel] = button[:,:,channel] + shift
    #button = button + 100
    img[box[1]:box[3]+1,box[0]:box[2]+1] = button
    return img

def change_size(img, box):
    button = img[box[1]:box[3]+1,box[0]:box[2]+1]
    height, width, channel = button.shape
    button = Image.fromarray(button.astype('uint8')).convert('RGB')
    #button.show()
    k = random.uniform(0.5, 3)
    new_width = int(width*k)
    new_height = int(height*k)
    button = button.resize((new_width, new_height), Image.ANTIALIAS)
    
    left = (new_width - width)/2
    top = (new_height - height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    #button.show()
    button = button.crop((left, top, right, bottom))
    button = button.resize((width, height), Image.ANTIALIAS)
    #button.show()
    img[box[1]:box[3]+1,box[0]:box[2]+1] = np.array(button)
    return img

def modify_images(train_batch):
    x_batch = []
    for i in train_batch:
        
        img = i[0].copy()
        box = i[1].copy()
        box = [int(x) for x in box]
        
        a = box[2] - box[0]
        b = box[3] - box[1]
        
        if (a < 0 or b < 0):
            chance = random.randint(0,0)
            print("Wrong button bound!!!")
        else:
            chance = random.randint(0,1)
        if (chance == 0):
            x_batch.append(change_color(img, box))
        else:
            x_batch.append(change_size(img, box))
    return x_batch

if __name__ == '__main__':
    test()
    
