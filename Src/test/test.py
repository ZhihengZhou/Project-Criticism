import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
import random
import pickle
from PIL import Image
sys.path.append('..')
from network256 import Network
from load import load_test, load
from collections import Counter


# Hyperparameters
IMAGE_SIZE = 256
LOCAL_SIZE = 64
BATCH_SIZE = 16

# Get test data
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
    
    # Setup Tensor
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    x_modified = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])

    # Create model
    model = Network(x, x_modified, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Load the model
    saver = tf.train.Saver()
    saver.restore(sess, '../backup/latest')

    # np.random.shuffle(test_data)
    
    step_num = int(len(test_data) / BATCH_SIZE) # Total amount of batches
    
    test_results = [] # Store results list

    cnt = 0 # Test results counter
    for step_index in tqdm.tqdm(range(step_num)):
        # Get test batch. (Init as 16)
        test_batch = test_data[step_index * BATCH_SIZE:(step_index + 1) * BATCH_SIZE]
        
        # Get original image and normalised
        x_batch = np.array([i[0] for i in test_batch])
        x_batch = np.array([a / 127.5 - 1 for a in x_batch])
        
        # Get modified image and normalised
        x_batch_modified = modify_images(test_batch)
        x_batch_modified = np.array([a / 127.5 - 1 for a in x_batch_modified])
        
        # Get modified area bounds and masks
        bounds = np.array([i[1] for i in test_batch])
        _, mask_batch = get_points(bounds)
        
        # Get other components bound
        other_bounds = np.array([i[2] for i in test_batch])
        
        # Run the model
        completion = sess.run(model.imitation, feed_dict={x: x_batch, x_modified: x_batch_modified, mask: mask_batch, is_training: False})
        
        # Test results in batch
        for batch_index in range(BATCH_SIZE):
            # print(batch_index)
            cnt += 1
            
            # Original image
            raw = x_batch[batch_index]
            raw = np.array((raw + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./real/{}.jpg'.format("{0:06d}".format(cnt)), raw)
            
            # Modified image
            # masked = raw * (1 - mask_batch[batch_index]) + np.ones_like(raw) * mask_batch[batch_index] * 255
            modified = x_batch_modified[batch_index]
            modified = np.array((modified + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./input/{}.jpg'.format("{0:06d}".format(cnt)), modified)
            
            # Model output image
            img = completion[batch_index]
            img = np.array((img + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(cnt)), img)
            
            # Get original mask
            original_mask = mask_batch[batch_index]
            original_mask = original_mask == 1
            original_mask = np.reshape(original_mask, (256,256))
            mask_num = np.sum(original_mask)
            
            # Get delta mask (abs(input - output))
            in_int = np.array(modified, dtype=int)
            out_int = np.array(img, dtype=int)
            
            delta = in_int - out_int
            delta = abs(delta)
            delta = delta[:,:,0] + delta[:,:,1] + delta[:,:,2]

            test_results.append((delta, bounds[batch_index], other_bounds[batch_index]))
            
            dst = './aggregate/{}.jpg'.format("{0:06d}".format(cnt))
            # cv2.imwrite('./aggregate/{}.jpg'.format("{0:06d}".format(cnt)), delta/3)
            output_image([['Input', modified], ['Output', img], ['Ground Truth', raw], ['Mask', delta]], dst, bounds[batch_index])

    np.save("test_results.npy", test_results)


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
    fig = plt.figure(figsize=(15,4))
    for i, image in enumerate(images):
        text, img = image
        fig.add_subplot(1, len(images), i + 1)
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
    shift = random.randint(50, 150)
    if random.randint(0, 1) == 1:
        shift *= -1
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
            chance = random.randint(0,0)
        if (chance == 0):
            x_batch.append(change_color(img, box))
        else:
            x_batch.append(change_size(img, box))
    return x_batch

if __name__ == '__main__':
    test()
    
