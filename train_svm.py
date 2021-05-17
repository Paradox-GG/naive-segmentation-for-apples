import cv2
import random
import numpy as np
import os
from sklearn.svm import SVC
import joblib


def img_split(img: np.ndarray):
    foreground = list()
    white_background_threshold = 720

    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            pixel = img[m][n]
            if np.sum(pixel) >= white_background_threshold:
                pass
            else:
                foreground.append(img[m][n])

    return foreground


def estimate3D(pixels):
    step = 16
    split = np.arange(0, 255, step)
    hist = np.zeros((len(split), len(split), len(split)))

    for pixel in pixels:
        hist[pixel[0]//step, pixel[1]//step, pixel[2]//step] += 1

    return hist


def pixel_generator(prob, pc, step=16):
    index = 0
    while prob > pc[index]:
        index += 1
    # (m, n, k)
    m, nn = divmod(index, step*step)
    n, k = divmod(nn, step)

    c1 = random.randint(step * m, step * (m + 1) - 1)
    c2 = random.randint(step * n, step * (n + 1) - 1)
    c3 = random.randint(step * k, step * (k + 1) - 1)
    return np.array([c1, c2, c3])


def generate_background_pixels(hist: np.ndarray, bnum):
    background_pixels = list()

    bhist = np.max(hist) - hist.reshape(-1)

    prob_cum = np.cumsum(bhist)
    prob_cum = prob_cum / prob_cum[-1]

    for n in range(bnum):
        random_choice = random.random()
        bp = pixel_generator(random_choice, prob_cum)
        background_pixels.append(bp)
    return background_pixels


def img_resize(img, h_t=400, w_t=300):
    if img.shape[0] > h_t:
        img = cv2.resize(src=img, dsize=(h_t, int(h_t * img.shape[1] / img.shape[0])))
    if img.shape[1] > w_t:
        img = cv2.resize(src=img, dsize=(int(w_t * img.shape[0] / img.shape[1]), w_t))
    return img


def pic2list(img):
    piclist = list()
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            piclist.append(img[m][n])
    return piclist


def list2pic(pl, img):
    mask = np.zeros_like(img)
    for m in range(mask.shape[0]):
        for n in range(mask.shape[1]):
            mask[m][n][0] = pl[m * mask.shape[1] + n]
            mask[m][n][1] = pl[m * mask.shape[1] + n]
            mask[m][n][2] = pl[m * mask.shape[1] + n]
    return mask * 255


def train(train_path, mode_save=False):
    train_pic_names = os.listdir(train_path)

    all_foreground_pixels = list()
    for pic_n in train_pic_names:
        train_pic_path = os.path.join(train_path, pic_n)
        pic = cv2.imread(train_pic_path)
        pic = img_resize(pic)
        single_foreground_pixels = img_split(pic)
        all_foreground_pixels += single_foreground_pixels
    random.shuffle(all_foreground_pixels)

    max_pixel_num = 1000
    if len(all_foreground_pixels) >= max_pixel_num:
        sampled_foreground_pixels = random.sample(all_foreground_pixels, max_pixel_num)

    else:
        sampled_foreground_pixels = all_foreground_pixels

    print('start estimate the distribution...')
    hist = estimate3D(all_foreground_pixels)
    print('finish.')

    background_pixels = generate_background_pixels(hist, len(sampled_foreground_pixels))

    foreground_labels = len(sampled_foreground_pixels) * [1]
    background_labels = len(background_pixels) * [0]

    pixels = sampled_foreground_pixels + background_pixels
    labels = foreground_labels + background_labels

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    svm = SVC(kernel=kernels[0], probability=True)
    print('start training svm, kernel is {}.'.format(kernels[0]))
    svm.fit(pixels, labels)
    print('finish.')

    if mode_save:
        print('saving the model.')
        joblib.dump(svm, 'apple_svm_1.model')


if __name__ == "__main__":
    train_path = 'imgs_apple/train/'
    train(train_path=train_path, mode_save=False)
