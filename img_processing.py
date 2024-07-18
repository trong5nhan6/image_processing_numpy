import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_gray_img(gray_img):
    plt.imshow(gray_img, cmap='gray')
    plt.show()


def show_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def convert_to_gray(img):
    return np.dot(img[..., :3], [0.21, 0.72, 0.07])


def bright_img(img, k):
    img_plus_k = img.astype(np.int32) + k
    new_img = np.clip(img_plus_k, 0, 255).astype(np.uint8)
    return new_img


def compute_difference(bg_img, input_img):
    difference_single_channel = cv2.absdiff(bg_img, input_img)
    return difference_single_channel


def compute_binary_mask(difference_single_channel):
    difference_binary = np.where(difference_single_channel < 10, 0, 255)
    return difference_binary


def replace_background(bg1_image, bg2_image, ob_image):
    difference_single_channel = compute_difference(bg1_image, ob_image)
    binary_mask = compute_binary_mask(difference_single_channel)
    output = np.where(binary_mask == 255, ob_image, bg2_image)
    return output
