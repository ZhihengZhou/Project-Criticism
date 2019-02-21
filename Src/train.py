import os
import cv2
import numpy as np
import tqdm
import tensorflow as tf
import random
from layer import *
from network256 import *
from load import *
from PIL import Image

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

# def change_size(img, box):
#     button = img[box[1]:box[3]+1,box[0]:box[2]+1]
#     height, width, channel = button.shape
#     k = random.uniform(0.5, 3)
#     new_width = int(width*k)
#     new_height = int(height*k)
#     button = cv2.resize(button, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
#     left = int((new_width - width)/2)
#     top = int((new_height - height)/2)
#     right = int((width + new_width)/2)
#     bottom = int((height + new_height)/2)

#     button = button[top:bottom, left:right].copy()

#     button = cv2.resize(button, (width, height), interpolation=cv2.INTER_CUBIC)
    
#     img[box[1]:box[3]+1,box[0]:box[2]+1] = np.array(button)
#     return img
    

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
        else:
            chance = random.randint(0,0)
        if (chance == 0):
            x_batch.append(change_color(img, box))
        else:
            x_batch.append(change_size(img, box))
    return x_batch

# Hyperparameters
IMAGE_SIZE = 256
LOCAL_SIZE = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
PRETRAIN_EPOCH = 50

x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
x_modified = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
is_training = tf.placeholder(tf.bool, [])

model = Network(x, x_modified, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)

global_step = tf.Variable(0, name='global_step', trainable=False)
epoch = tf.Variable(0, name='epoch', trainable=False)

opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)

d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# Load train and test data.
train_data, test_data = load()

print(len(train_data))
train_data = [x for x in train_data if len(x[1]) == 4]
print(len(train_data))

train_data = [x for x in train_data if x[0].shape == (256,256,3)]
print(len(train_data))

print(len(test_data))
test_data = [x for x in test_data if len(x[1]) == 4]
if len(test_data) < BATCH_SIZE:
    test_data = train_data
print(len(test_data))

test_data = [x for x in test_data if x[0].shape == (256,256,3)]
print(len(test_data))

step_num = int(len(train_data) / BATCH_SIZE)

# Load model
if tf.train.get_checkpoint_state('./backup'):
    saver = tf.train.Saver()
    saver.restore(sess, './backup/latest')

while True:
    sess.run(tf.assign(epoch, tf.add(epoch, 1)))
    print('epoch: {}'.format(sess.run(epoch)))
    
    np.random.shuffle(train_data)
    
    # Completion
    if sess.run(epoch) <= PRETRAIN_EPOCH:
        g_loss_value = 0
        for i in tqdm.tqdm(range(step_num)):
            train_batch = train_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            
            x_batch = np.array([i[0] for i in train_batch])
            x_batch = np.array([a / 127.5 - 1 for a in x_batch])
            
            x_batch_modified = modify_images(train_batch)
            x_batch_modified = np.array([a / 127.5 - 1 for a in x_batch_modified])    
            
            points_batch, mask_batch = get_points([i[1] for i in train_batch])
            
            _, g_loss = sess.run([g_train_op, model.g_loss], feed_dict={x: x_batch, x_modified: x_batch_modified, mask: mask_batch, is_training: True})
            g_loss_value += g_loss
    
        print('Completion loss: {}'.format(g_loss_value))
        
        np.random.shuffle(test_data)
        test_batch = test_data[:BATCH_SIZE]
        
        x_batch = np.array([i[0] for i in test_batch])
        x_batch = np.array([a / 127.5 - 1 for a in x_batch])
        
        x_batch_modified = modify_images(test_batch)
        x_batch_modified = np.array([a / 127.5 - 1 for a in x_batch_modified])
        
        points_batch, mask_batch = get_points([i[1] for i in test_batch])
        
        completion = sess.run(model.imitation, feed_dict={x: x_batch, x_modified: x_batch_modified, mask: mask_batch, is_training: False})
        sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
        cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
        
        saver = tf.train.Saver()
        saver.save(sess, './backup/latest', write_meta_graph=False)
        if sess.run(epoch) == PRETRAIN_EPOCH:
            saver.save(sess, './backup/pretrained', write_meta_graph=False)

    # Discrimitation
    else:
        g_loss_value = 0
        d_loss_value = 0
        for i in tqdm.tqdm(range(step_num)):
            train_batch = train_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            x_batch = np.array([i[0] for i in train_batch])
            x_batch = np.array([a / 127.5 - 1 for a in x_batch])
            
            x_batch_modified = modify_images(train_batch)
            x_batch_modified = np.array([a / 127.5 - 1 for a in x_batch_modified])
            
            points_batch, mask_batch = get_points([i[1] for i in train_batch])
            
            _, g_loss, completion = sess.run([g_train_op, model.g_loss, model.completion], feed_dict={x: x_batch, x_modified: x_batch_modified, mask: mask_batch, is_training: True})
            g_loss_value += g_loss
            
            local_x_batch = []
            local_completion_batch = []
            for i in range(BATCH_SIZE):
                x1, y1, x2, y2 = points_batch[i]
                local_x_batch.append(x_batch[i][x1:x2, y1:y2, :])
                local_completion_batch.append(completion[i][x1:x2, y1:y2, :])
            local_x_batch = np.array(local_x_batch)
            local_completion_batch = np.array(local_completion_batch)
            
            _, d_loss = sess.run(
                                 [d_train_op, model.d_loss],
                                 feed_dict={x: x_batch, x_modified: x_batch_modified, mask: mask_batch, local_x: local_x_batch, global_completion: completion, local_completion: local_completion_batch, is_training: True})
            d_loss_value += d_loss
        
        print('Completion loss: {}'.format(g_loss_value))
        print('Discriminator loss: {}'.format(d_loss_value))
        
        np.random.shuffle(test_data)
        test_batch = test_data[:BATCH_SIZE]
        
        x_batch = np.array([i[0] for i in test_batch])
        x_batch = np.array([a / 127.5 - 1 for a in x_batch])
        
        x_batch_modified = modify_images(test_batch)
        x_batch_modified = np.array([a / 127.5 - 1 for a in x_batch_modified])
        
        points_batch, mask_batch = get_points([i[1] for i in test_batch])
        
        completion = sess.run(model.imitation, feed_dict={x: x_batch, x_modified: x_batch_modified, mask: mask_batch, is_training: False})
        sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
        cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
        
        saver = tf.train.Saver()
        saver.save(sess, './backup/latest', write_meta_graph=False)

#if __name__ == '__main__':
#    x_train, x_test = load()
#    print(x_train.shape)
#    print(x_test.shape)

