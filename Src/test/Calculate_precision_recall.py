import numpy as np
from collections import Counter
import cv2
import tqdm

test_results = np.load("test_results.npy")

# Hyperparameters
diff_threshold = 0
predict_threshold = 0.8

IoU = []
recall = []
precision = []
threshold_list = []

for result in tqdm.tqdm(test_results):
    delta = result[0]
    target_bound = result[1]
    other_bounds = result[2]
    
    # Calucuate pixel diff threshold
    m_list = delta.flatten()
    h_dic = dict(Counter(m_list))
# #     for i in range(max(h_dic.keys()),-1,-1):
# #         if i in h_dic.keys() and h_dic[i] > 64*64: # 100, 500, 1000
# #             threshold = i
# #             break
            
#     pixel_sum = 0
#     threshold = 0
#     # mask_num = (target_bound[2] - target_bound[0] + 1) * (target_bound[3] - target_bound[1] + 1)
#     mask_num = 10000
#     for i in range(max(h_dic.keys()),-1,-1):
#         if i in h_dic.keys():
#             pixel_sum += h_dic[i]
#             if pixel_sum > mask_num:
#                 threshold = i
#                 break

    
    threshold_percent = 0.8
    sum_threshold = threshold_percent*256*256
    pixel_sum = 0
    for i in range(max(h_dic.keys())):
        if i in h_dic.keys():
            pixel_sum += h_dic[i]
            if pixel_sum > sum_threshold:
                threshold = i
                break
    threshold_list.append(threshold)
    # Get original mask
    original_mask = np.zeros((delta.shape[1], delta.shape[0]))
    original_mask[target_bound[1]:target_bound[3]+1, target_bound[0]:target_bound[2]+1] = 1
    original_mask = original_mask == 1
    
    # Calculate IoU
    change_mask = delta > threshold
    change_num = np.sum(change_mask)
    intersection = np.sum(original_mask * change_mask)
    union = np.sum(original_mask + change_mask)
    IoU.append(intersection/union)
    
    # Calculate components
    target_component = change_mask[target_bound[1]:target_bound[3]+1, target_bound[0]:target_bound[2]+1]
    other_components = []
    for bound in other_bounds:
        other_components.append(change_mask[bound[1]:bound[3]+1, bound[0]:bound[2]+1])
        
    # Calculate recall
    target_percentage = np.sum(target_component)/target_component.size
    if target_percentage >= predict_threshold:
        recall_value = 1
    else:
        recall_value = 0
    recall.append(recall_value)
    
    # Calculate precision
    predict_count = recall_value
    for component in other_components:
        component_percentage = np.sum(component)/component.size
        if component_percentage >= predict_threshold:
            predict_count += 1
    if predict_count == 0:
        precision.append(0)
        continue
    precision.append(recall_value/predict_count)
    
print(np.sum(IoU)/len(IoU))
print(np.sum(recall)/len(recall))
print(np.sum(precision)/len(precision))
print(np.sum(threshold_list)/len(threshold_list))

np.save("precesion.npy", (IoU, recall, precision))