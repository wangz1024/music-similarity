"""
    生成特征向量
"""

from keras.models import load_model
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import pymysql
import csv

model = load_model('model.h5')
data_dir = '/home/wangz/Desktop/音乐推荐系统/train_data'
spectrum_dir_list = glob.glob(os.path.join(data_dir, '*'))
spectrum_list = glob.glob(os.path.join(data_dir, '*/*.jpg'))


# 图片预处理
def img_preprocess(image):
    img = image.astype('float32') / 255
    img = img - img.mean()
    img = img / img.std(axis=0)
    # 提升维度(batch_size=1)
    img = np.expand_dims(img, axis=0)
    # print(img.shape)
    return img

# for spec_path in spectrum_list:
#     image = plt.imread(spec_path)
#     plt.close()
#     image = img_preprocess(image)
#     predictions = model.predict(image)
#     print(spec_path.split('/')[-1].replace('.jpg', ''), spec_path.split('/')[-2], '   =======>   ', \
#           [round(x, 2) if x > 0.01 else 0 for x in predictions[0]])
#
#     # print classify
#     # print(spec_path.split('/')[-1].replace('.jpg', ''), spec_path.split('/')[-2], '   =======>   ', \
#     #       np.argmax(predictions, axis=1))

# 导出为txt文件
# for dir_path in spectrum_dir_list:
#     files = os.listdir(dir_path)
#     files.sort()
#     vector_list = []
#     current_classfication = dir_path.split('/')[-1]
#     current_txt_path = os.path.join('/home/wangz/Desktop/音乐推荐系统/vector', current_classfication+'.txt')
#     for file in files:
#         image = plt.imread(os.path.join(dir_path, file))
#         image = img_preprocess(image)
#         predictions = model.predict(image)
#         vector_list.append([round(x, 2) if x > 0.01 else 0 for x in predictions[0]])
#     with open(current_txt_path, 'w+') as f:
#         for name, vec in zip(files, vector_list):
#              f.write(name.replace('.jpg', '')+','+'-'.join([str(x) for x in vec]))
#              f.write('\n')
#         f.close()

# 写入至数据库
db = pymysql.connect('localhost', 'root', '123abc', 'music_info')
cursor = db.cursor()

for dir_path in spectrum_dir_list:
    files = os.listdir(dir_path)
    files.sort()
    vector_list = []
    classfication = dir_path.split('/')[-1].lower()
    for file in files:
        image = plt.imread(os.path.join(dir_path, file))
        image = img_preprocess(image)
        predictions = model.predict(image)
        name = file.replace('.jpg', '')
        vector = '-'.join([str(x) for x in [round(x, 2) if x > 0.01 else 0 for x in predictions[0]]])
        sql = """INSERT INTO %s (name, vector) VALUES ('%s', '%s');""" % (classfication, name, vector)
        try:
            cursor.execute(sql)
            db.commit()
        except:
            print(sql)
            db.rollback()

db.close()









