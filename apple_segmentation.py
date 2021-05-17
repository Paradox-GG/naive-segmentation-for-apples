import cv2
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt


def draw_a_pic(pic, no):
    plt.figure(no)
    plt.imshow(pic)
    plt.show()


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
            mask[m, n, :] = int(255 * pl[m * mask.shape[1] + n])
    return mask


def watershed_alg(pred, img, no):
    save_or_show = False

    ret, thresh = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 对前景区域膨胀
    sure_bg = cv2.dilate(opening, kernel, iterations=20)

    # 获得测地线距离图并归一化
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    foreground_threshold = 0.3  # 0.5 if use gray
    ret, sure_fg = cv2.threshold(dist_transform, foreground_threshold * dist_transform.max(), 255, 0)  # threshold
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # 对连通区域标记
    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0
    # draw_a_pic(markers, 3)

    # 调用分水岭算法， 蓝色边界用蓝色线标记
    markers = cv2.watershed(img, markers)
    # draw_a_pic(markers, 4)

    img[markers == -1] = [255, 0, 0]

    if save_or_show:
        file_name = 'ar{}.png'.format(no)
        cv2.imwrite(file_name, img)
    else:
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def segment(test_path, model_path):
    test_pic_names = os.listdir(test_path)
    svm = joblib.load(model_path)

    no = 1
    for pic_n in test_pic_names:
        test_pic_path = os.path.join(test_path, pic_n)
        pic = cv2.imread(test_pic_path)
        test_pixels = pic2list(pic)

        pred_p = svm.predict_proba(test_pixels)
        pred_p = pred_p[:, 0]
        pred_p_mask = list2pic(pred_p, pic)
        watershed_alg(pred_p_mask[:, :, 0], img=pic, no=no)
        no += 1

    # for pic_n in test_pic_names:
    #     test_pic_path = os.path.join(test_path, pic_n)
    #     pic = cv2.imread(test_pic_path)
    #     gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    #     watershed_alg(gray, img=pic, no=no)
    #     no += 1


if __name__ == '__main__':
    test_path = 'imgs_apple/test/'
    model_path = 'apple_svm.model'
    segment(test_path=test_path, model_path=model_path)
