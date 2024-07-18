import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from effect_img import *


def apply_effect(image_effects, effect_name, img):
    if effect_name == 'adjust_contrast':
        return adjust_contrast(img)
    elif effect_name == 'hdr':
        return hdr(img)
    elif effect_name == 'smooth_skin':
        return smooth_skin(img)
    elif effect_name == 'bokeh':
        return bokeh(img)
    elif effect_name == 'vintage':
        return vintage(img)
    elif effect_name == 'cartoon':
        return cartoon(img)
    elif effect_name == 'sketch':
        return sketch(img)
    elif effect_name == 'oil_painting_alternative':
        return oil_painting_alternative(img)
    elif effect_name == 'polarizing':
        return polarizing(img)
    elif effect_name == 'thermal':
        return thermal(img)
    elif effect_name == 'pointillism':
        return pointillism(img)
    elif effect_name == 'glitch':
        return glitch(img)
    elif effect_name == 'comic_book':
        return comic_book(img)
    elif effect_name == 'artistic':
        return artistic(img)
    elif effect_name == 'gaussian_blur':
        return gaussian_blur(img)
    elif effect_name == 'artistic_filter':
        return artistic_filter(img)
    elif effect_name == 'Gaussian_noise':
        return Gaussian_noise(img)
    elif effect_name == 'unique_style':
        return unique_style(img)
    elif effect_name == 'random_style':
        return random_style(img)
    else:
        return img


def img_eff_flow(img):
    # Danh sách các hiệu ứng hình ảnh
    image_effects = [
        'random_style', 'gaussian_blur', 'Gaussian_noise', 'unique_style',
        'cartoon', 'comic_book', 'sketch', 'artistic',
        'hdr', 'smooth_skin', 'thermal', 'vintage',
        'bokeh', 'glitch', 'artistic_filter', 'polarizing',
        'pointillism', 'oil_painting'
    ]

    st.subheader('Select Effects:')
    check_boxs = []
    num_columns = 4

    for i in range(0, len(image_effects), num_columns):
        cols = st.columns(num_columns)
        for j, col in enumerate(cols):
            if i + j < len(image_effects):
                check_box = col.checkbox(image_effects[i + j])
                check_boxs.append((check_box, image_effects[i + j]))

    for check_box, effect_name in check_boxs:
        if check_box:
            result_image = apply_effect(image_effects, effect_name, img)
            st.image(result_image,
                     caption=f'Effect: {effect_name}', use_column_width=True)


def main():
    # Tạo cột cho checkbox và hình ảnh kết quả
    st.title('Image Effect App with Streamlit')

    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Hiển thị ảnh đã tải lên
        img = Image.open(uploaded_image)
        img = np.asarray(img, dtype=np.uint8)
        st.image(img, caption='Original Image', use_column_width=True)

        img_eff_flow(img)


if __name__ == '__main__':
    main()
