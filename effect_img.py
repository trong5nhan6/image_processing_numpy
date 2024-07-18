import numpy as np
import cv2
import matplotlib.pyplot as plt


def adjust_contrast(image, factor=1.5):
    # Tăng cường độ tương phản (mỗi pixel nhân với factor)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def hdr(image):
    # Áp dụng hiệu ứng HDR
    hdr = cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)
    return hdr


def smooth_skin(image):
    # Làm mịn da
    return cv2.bilateralFilter(image, 50, 75, 75)


def bokeh(image, blur_radius=21):
    # Làm mờ nền
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)
    bokeh = cv2.addWeighted(
        image, 0.75, cv2.merge([mask, mask, mask]), 0.25, 0)
    return bokeh


def vintage(image):
    # Thêm hiệu ứng vintage
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    vintage = cv2.transform(image, sepia_filter)
    vintage = np.clip(vintage, 0, 255)
    return vintage.astype(np.uint8)


def cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv_gray = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    inv_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray, inv_blur, scale=256.0)
    return sketch


def oil_painting_alternative(image):
    dst = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            roi = image[max(0, y - 4):min(y + 5, image.shape[0]),
                        max(0, x - 4):min(x + 5, image.shape[1])]
            dst[y, x] = np.median(roi, axis=(0, 1))
    return dst


def polarizing(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = cv2.equalizeHist(hsv_image[:, :, 1])
    polarizing_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return polarizing_image


def thermal(image):
    thermal = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return thermal


def pointillism(image):
    pointillism = cv2.stylization(image, sigma_s=60, sigma_r=0.3)
    return pointillism


def glitch(image):
    rows, cols, _ = image.shape
    glitch = image.copy()
    num_lines = 10
    for _ in range(num_lines):
        y1 = np.random.randint(0, rows)
        y2 = np.random.randint(y1, rows)
        x_shift = np.random.randint(-20, 20)
        glitch[y1:y2, :] = np.roll(glitch[y1:y2, :], x_shift, axis=1)
    return glitch


def comic_book(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    comic = cv2.bitwise_and(color, color, mask=edges)
    return comic


def artistic(image):
    artistic = cv2.edgePreservingFilter(
        image, flags=1, sigma_s=60, sigma_r=0.4)
    return artistic


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)


def artistic_filter(image):
    # Chuyển đổi ảnh sang định dạng HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Thay đổi kênh Hue để tạo ra hiệu ứng màu sắc độc đáo
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + 50) % 180

    # Chuyển đổi lại sang định dạng BGR
    artistic_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return artistic_image


def Gaussian_noise(image):
    # Thêm nhiễu Gaussian vào ảnh
    noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    return noisy_image


def create_pattern(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    patterned_image = cv2.filter2D(image, -1, kernel)
    return patterned_image


def unique_style(image):
    # Thêm nhiễu vào ảnh
    noisy_image = Gaussian_noise(image)

    # Tạo hoa văn độc đáo
    final_image = create_pattern(noisy_image)
    return final_image


def random_style(image):
    # Generate random values in the range [-1, 1]
    kernel = np.random.uniform(low=0, high=1, size=(3, 3))
    random_img = cv2.transform(image, kernel)
    random_img = np.clip(random_img, 0, 255)
    return random_img.astype(np.uint8)
