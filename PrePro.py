"""
    频谱图预处理
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob

spectrum_input_dir = '/home/wangz/Desktop/音乐推荐系统/ballroomMelSpec'
spectrum_output_dir = '/home/wangz/Desktop/音乐推荐系统/train_data'

spectrum_list = glob.glob(os.path.join(spectrum_input_dir, '*/*.jpg'))
if not os.path.exists(spectrum_output_dir):
    os.mkdir(spectrum_output_dir)
#
#图像处理
with tf.Session() as sess:
    for image_path in spectrum_list:
        raw_image = tf.gfile.GFile(image_path, 'rb').read()
        image = tf.image.decode_jpeg(raw_image)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        resize_image = sess.run(tf.image.resize_images(image, (128, 128), method=0))
        # 颜色特征保留，无必要灰度化
        # gray_image = sess.run(tf.image.rgb_to_grayscale(resize_image))
        # fig = plt.figure()
        # plt.axis('off')
        # plt.imshow(gray_image[:,:,0], cmap='gray')
        dir_path = os.path.join(spectrum_output_dir, image_path.split('/')[-2])
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        save_path = os.path.join(dir_path, os.path.basename(image_path))
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=-1)
        # plt.close(fig)
        plt.imsave(save_path, resize_image)
        # break

# image = plt.imread('/home/wangz/Desktop/音乐推荐系统/train_data/Tango/000002.jpg')
# print(image.shape)
