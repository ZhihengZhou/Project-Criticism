import os
import numpy as np

def load(dir_='../Data/UIdata/npy-Crop/'):
    files = os.listdir(dir_)
    train_files = [x for x in files if "train" in x]
    test_files = [x for x in files if "test" in x]
    
    x_train = []
    x_test = []
    
    for f in train_files:
        x_train.append(np.load(os.path.join(dir_, f)))
    for f in test_files:
        x_test.append(np.load(os.path.join(dir_, f)))
    
    return np.vstack(x_train), np.vstack(x_test)

def load_test(dir_='../../Data/UIdata/npy-Crop/'):
    files = os.listdir(dir_)
    
    test_files = [x for x in files if "test" in x]
    
    x_test = []
    
    for f in test_files:
        x_test.append(np.load(os.path.join(dir_, f)))
    
    return np.vstack(x_test)

