import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
import math

data_dir = './Data/stage1/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('./Data/stage1_labels.csv', index_col='id')
submit_df = pd.read_csv('./Data/stage1_sample_submission.csv', index_col='id')

IMG_SIZE_PX = 50
SLICE_COUNT = 20

# def chunks(l, n):
#     # Credit: Ned Batchelder
#     # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i + n]

# def mean(l):
#     return sum(l) / len(l)

# def process_data(patient,labels_df,img_px_size=50, hm_slices=20, visualize=False):
    
#     label = labels_df.at[patient, 'cancer']
#     path = data_dir + patient
#     slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
#     slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

#     new_slices = []
#     slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]
    
#     chunk_sizes = math.ceil(len(slices) / hm_slices)
#     for slice_chunk in chunks(slices, chunk_sizes):
#         slice_chunk = list(map(mean, zip(*slice_chunk)))
#         new_slices.append(slice_chunk)

#     if len(new_slices) == hm_slices-1:
#         new_slices.append(new_slices[-1])

#     if len(new_slices) == hm_slices-2:
#         new_slices.append(new_slices[-1])
#         new_slices.append(new_slices[-1])

#     if len(new_slices) == hm_slices+2:
#         new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
#         del new_slices[hm_slices]
#         new_slices[hm_slices-1] = new_val
        
#     if len(new_slices) == hm_slices+1:
#         new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
#         del new_slices[hm_slices]
#         new_slices[hm_slices-1] = new_val

#     if visualize:
#         fig = plt.figure()
#         for num,each_slice in enumerate(new_slices):
#             y = fig.add_subplot(4,5,num+1)
#             y.imshow(each_slice, cmap='gray')
#         plt.show()

#     if label == 1: label=np.array([0,1])
#     elif label == 0: label=np.array([1,0])
        
#     return np.array(new_slices), label

# much_data = []
# for num, patient in enumerate(patients):
#     if num % 100 == 0:
#         print(num)
#     try:
#         img_data,label = process_data(patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
#         #print(img_data.shape,label)
#         much_data.append([img_data,label])
#     except KeyError as e:
#         print('This is unlabeled data!')

# np.save('muchdata-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), much_data)



target = 1
print(labels_df.head(target))
print(labels_df.shape)
for patient in patients[:target]:
    try: # 有些 patient 沒有在labels.df
        label = labels_df.at[patient, 'cancer']
        path = data_dir + patient
        
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

        slices.sort(key = lambda x : int(x.ImagePositionPatient[2]))
        print(slices[0])
    except:
        pass

#         # print 一下pydicom整理的資訊
#         # for num, each_slice in enumerate(slices):
#         #     print('No:', num)
#         #     print(each_slice)
#         # 1. Check images size
#         # print(slices[0].pixel_array.shape, len(slices))
#         # print('ImagePositionPatient length:')
#         # print(len(slices[0].ImagePositionPatient))
#         # print('slices 0 ImagePositionPatient[0]:')
#         # print(slices[0].ImagePositionPatient[0])    
#         # print('slices 0 ImagePositionPatient[1]:')
#         # print(slices[0].ImagePositionPatient[1])
#         # print('slices 0 ImagePositionPatient[2]:')
#         # print(slices[0].ImagePositionPatient[2])

#         # print('Patient: {}, DCM slice數: {}, Cancer status: {}'.format(patient, len(slices), label))

#         # 2. Check slice image
#         # print('show slice 0:')
#         # print(slices[0])
#         # plt.imshow(slices[0].pixel_array)
#         # plt.show()

#         # 3. Check more slice from a patient (用 openCV resize 圖片)
#         # IMG_PX_SIZE = 150
#         # fig = plt.figure()
#         # for num, each_slice in enumerate(slices[:12]):
#         #     y = fig.add_subplot(3, 4, num + 1)
#         #     new_img = cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE))
#         #     y.imshow(new_img)
#         # plt.show()

#         # 4. 調整所有scan的depth到一致

#         new_slices = []
#         slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]  
#         chunk_sizes = math.ceil(len(slices) / HM_SLICES)
#         for slice_chunk in chunks(slices, chunk_sizes):
#             slice_chunk = list(map(mean, zip(*slice_chunk)))
#             new_slices.append(slice_chunk)

#         # print(len(slices), len(new_slices))

#         # 它上面沒有處理好，有些patient depth 20，有些18, 19, 21, 22 等等都有
#         if len(new_slices) == HM_SLICES-1:
#             new_slices.append(new_slices[-1])

#         if len(new_slices) == HM_SLICES-2:
#             new_slices.append(new_slices[-1])
#             new_slices.append(new_slices[-1])

#         if len(new_slices) == HM_SLICES+2:
#             new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
#             del new_slices[HM_SLICES]
#             new_slices[HM_SLICES-1] = new_val
            
#         if len(new_slices) == HM_SLICES+1:
#             new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
#             del new_slices[HM_SLICES]
#             new_slices[HM_SLICES-1] = new_val
        
#         fig = plt.figure()
#         for num,each_slice in enumerate(new_slices):
#             y = fig.add_subplot(4,5,num+1)
#             y.imshow(each_slice, cmap='gray')
#         plt.show()

#     except:
#         # 有些 patient 沒有在labels.df
#         print('exception')
#         pass




