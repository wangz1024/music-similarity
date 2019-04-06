"""
    生成梅尔频谱
"""

import librosa.display
import matplotlib.pyplot as plt
import os
import glob

ROOT_PATH = 'D:\\ballroom\\'
OUT_PATH = 'D:\\ballroomMelSpec\\'
counts = {}


def genMelSpec(music_path, type_name, time):
    count = counts.get(type_name, 1)
    counts[type_name] = count + 1

    dir_path = os.path.join(OUT_PATH, type_name)
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    y, sr = librosa.load(music_path, offset=time, duration=10)
    melspec = librosa.feature.melspectrogram(y, sr)
    logspec = librosa.power_to_db(melspec)
    fig = plt.figure()
    plt.axis('off')
    librosa.display.specshow(logspec, sr=sr)
    save_path = os.path.join(dir_path, type_name + "{}.jpg".format(str(count).zfill(6)))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# librosa.display.specshow(logspec, x_axis='time', y_axis='mel', sr=sr)
# plt.title('Mel-Spectrum')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()

if __name__ == '__main__':
    all_music_list = glob.glob(os.path.join(ROOT_PATH, '*\\*.wav'))
    for i, music_path in enumerate(all_music_list):
        type_name = music_path.split('\\')[-2]
        genMelSpec(music_path, type_name, 0)
        genMelSpec(music_path, type_name, 10)
        genMelSpec(music_path, type_name, 20)
        print('Now Execute ==>', i + 1, '/', len(all_music_list))
