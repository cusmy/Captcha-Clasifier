import numpy as np
import pandas as pd
import cv2 as cv
import pickle
import warnings
from itertools import product, permutations, combinations_with_replacement, chain
from math import floor
import streamlit as st
warnings.filterwarnings('ignore')

data = np.load('preprocessing-data.npz')
X, y_labels, y = data['X'], data['y_labels'], data['y']
with open('contour-classifier', 'rb') as file:
    contour_classifier = pickle.load(file)
with open('contour-classifier-preprocessor', 'rb') as file:
    contour_features_scaler = pickle.load(file)
def split_array(a, separators, axis=1):
    # Ini adalah fungsi pembantu untuk membagi array numpy di sepanjang sumbu yang diberikan menggunakan pemisah
    seperators = sorted(separators)
    n_sep = len(separators)
    if n_sep == 1:
        sep = separators[0]
        a = a.swapaxes(0, axis)
        return [a[0:sep].swapaxes(0, axis), a[sep:].swapaxes(0, axis)]

    head, body = split_array(a, [separators[0]], axis)
    splits = split_array(body, np.array(separators[1:]) - separators[0], axis)
    return [head] + splits

def find_separators(frame, n):
    # Metode ini mengembalikan n-1 garis vertikal dengan jarak yang sama untuk membagi frame yang ditunjukkan
    return np.floor(np.linspace(0, frame.shape[1], n+1)[1:-1]).astype(np.uint16)

def run_data(input):
    img = (X[input] * 255)[:, :, 0].astype(np.uint8)

    st.image(img)

    inverted = 255 - img

    st.write("Citra Negatif")

    st.image(inverted)

    ret, thresholded = cv.threshold(inverted, 140, 255, cv.THRESH_BINARY)

    st.write("Treshold")

    st.image(thresholded)

    blurred = cv.medianBlur(thresholded, 3)

    st.write("Mengurangi Noise")

    st.image(thresholded)

    kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]).astype(np.uint8)
    
    ex = cv.morphologyEx(blurred, cv.MORPH_OPEN, kernel)

    st.write("Transformasi Morfologi")

    st.image(ex)

    kernel2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
    ]).astype(np.uint8)
    ex2 = cv.morphologyEx(ex, cv.MORPH_DILATE, kernel2)

    st.image(ex2)

    mask = ex2
    processed = cv.bitwise_and(mask, blurred)
    st.image(processed)

    contours, hierachy = cv.findContours(processed, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = [contours[k] for k in range(0, len(contours)) if hierachy[0, k, 3] == -1]
    contours.sort(key=lambda cnt: cv.boundingRect(cnt)[0])


    st.image(cv.drawContours(cv.cvtColor(img, cv.COLOR_GRAY2RGB), contours, -1, (255, 0, 0), 1, cv.LINE_4), caption='Find Kontur')

    contour_bboxes = [cv.boundingRect(contour) for contour in contours]
    img_bboxes = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for bbox in contour_bboxes:
        left, top, width, height = bbox
        img_bboxes = cv.rectangle(img_bboxes,
                              (left, top), (left+width, top+height),
                              (0, 255, 0), 1)
    st.image(img_bboxes, caption='Box Kontur')

    contours_features = pd.DataFrame.from_dict({
    'bbox_width': [bbox[2] for bbox in contour_bboxes],
    'bbox_height': [bbox[3] for bbox in contour_bboxes],
    'area': [cv.contourArea(cnt) for cnt in contours],
    'extent': [cv.contourArea(cnt) / (bbox[2] * bbox[3]) for cnt, bbox in zip(contours, contour_bboxes)],
    'perimeter': [cv.arcLength(cnt, True) for cnt in contours]
    })
    st.write("Fitur pada Kontur")
    contours_features


    contour_features = contour_features_scaler.transform(contours_features[['bbox_width', 'bbox_height', 'area', 'extent', 'perimeter']])
    contour_num_chars = contour_classifier.predict(contour_features)

    n = len(contours)
    cols = 2
    rows = n // cols
    if n % cols > 0:
        rows += 1
    rows = max(rows, 2)

    for i, j in product(range(0,rows), range(0,cols)):
        k = i * cols + j
        if k < n:
            left, top, width, height = contour_bboxes[k]
            img_bbox = cv.rectangle(cv.cvtColor(img, cv.COLOR_GRAY2RGB),
                                (left, top), (left+width, top+height), (0, 255, 0), 1)
        
            st.write('Kontur {}, Jumlah Karakter: {}'.format(k, contour_num_chars[k]))
            st.image(img_bbox)
    P = contour_classifier.predict_proba(contour_features)
    configs = filter(lambda x: np.sum(x) == 5, combinations_with_replacement(range(0, 6), n))
    configs = list(frozenset(chain.from_iterable(map(lambda config: permutations(config, n), configs))))
    configs = np.array(configs, dtype=np.uint8)
    scores = np.zeros([len(configs)]).astype(np.float32)
    for i in range(0, len(configs)):
        scores[i] = np.prod(P[np.arange(0, n), configs[i]])
    best_config = configs[scores.argmax()]
    frames = []
    for i in range(0, n):
        if best_config[i] > 0:
            left, top, width, height = contour_bboxes[i]
            right, bottom = left+width, top+height
            frame = img[top:bottom, left:right]
            frames.append(frame)
    frame_num_chars = best_config[np.nonzero(best_config)[0]]
    num_frames = len(frames)

    cols = 3
    rows = num_frames // cols
    if num_frames % cols > 0:
        rows += 1
    rows = max(rows,2)

    for i, j in product(range(0,rows), range(0,cols)):
        k = i * cols + j
        if k < num_frames:
            st.write('Frame {}. Jumlah Karakter: {}'.format(k, frame_num_chars[k]))
            st.image(frames[k])
    frame = [frames[i] for i in range(0, num_frames) if frame_num_chars[i] > 1][-1]
    num_chars = frame_num_chars[frames.index(frame)]
    st.image(frame)


    chars = []
    for frame, num_chars in zip(frames, frame_num_chars):
        if num_chars == 1:
        # Tidak perlu membagi frame jika ada 1 char
            chars.append(frame)
        else:
        # membagi frame menjadi lebih dari 1 char
            splits = split_array(frame, find_separators(frame, num_chars), axis=1)
            chars.extend(splits)
        

    for i in range(0, 5):
        st.image(chars[i])
        st.write('Karakter {}, ukuran: {}'.format(i+1, chars[i].shape))

    chars_processed = np.zeros([5, 40, 40, 1]).astype(np.float32)
    for i, char in zip(range(0, 5), chars):
        img = char
        inverted = 255 - img
        ret, thresholded = cv.threshold(inverted, 70, 255, cv.THRESH_BINARY)
        img = 255 - np.multiply((thresholded > 0), inverted)
    
        dh, dw = 40, 40
        h, w = img.shape
        if w < dw:
            left = floor((dw - w) / 2)
            right = dw - w - left
            img = cv.copyMakeBorder(img, 0, 0, left, right, cv.BORDER_CONSTANT, value=(255, 255, 255))
        elif w > dw:
            left = floor((w - dw) / 2)
            img = img[:, left:left+dw]

        if h < dh:
            top = floor((dh - h) / 2)
            bottom = dh - h - top
            img = cv.copyMakeBorder(img, top, bottom, 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
        elif h > dh:
            top = floor((h - dh) / 2)
            img = img[top:top+dh, :]
    
        chars_processed[i, :, :, 0] = img.astype(np.float32) / 255

    for i in range(0, 5):
        st.image(chars_processed[i, :, :, 0])
        st.write('Karakter ke {}'.format(i+1))

input = st.slider('pilih index gambar, daripada ngetik mending geser geser', 0, 1070, 25)
btn = st.button("Proses")
if btn:
	run_data(input)
else:
	pass
