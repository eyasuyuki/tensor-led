import datetime
import itertools
import math

from keras import models
from imutils import contours
import cv2
import numpy as np

# from https://github.com/eyasuyuki/opencv-python-example
from keras.datasets import mnist
from keras.engine.saving import model_from_json
from keras.optimizers import Adam


def gray(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayed.jpg", grayed)
    return grayed


def denoise(img):
    denoised = cv2.fastNlMeansDenoising(img, 10, 10, 7, 21)
    cv2.imwrite("denoised.jpg", denoised)
    return denoised


def lut(img, min_table, max_table):
    diff_table = max_table - min_table
    lookup_table = np.arange(256, dtype="uint8")

    for i in range(0, len(lookup_table)):
        if i < min_table:
            lookup_table[i] = 0
        elif min_table <= i <= max_table:
            n = 255 * (i - min_table) / diff_table
            lookup_table[i] = n
        elif i > max_table:
            lookup_table[i] = 255

    contrast = cv2.LUT(img.copy(), lookup_table)
    cv2.imwrite("contrast.jpg", contrast)
    return contrast


def canny(img):
    edged = cv2.Canny(img, 20, 80, 255)
    cv2.imwrite("edged.jpg", edged)
    return edged


def filter_noise(img):
    # find contors

    cnts0 = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # filter noize

    thCnts = []
    for c in cnts0:
        (x, y, z, w) = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        # print(f"{x}, {y}, {z}, {w}, {a}")
        if a >= 10:
            thCnts.append(c)

    background = np.zeros_like(img, np.uint8)

    edged_cnts = cv2.drawContours(background, thCnts, -1, (255, 255, 255), 1)
    cv2.imwrite("edged_cnts.jpg", edged_cnts)
    return edged_cnts


def closing(img):
    # https://github.com/DevashishPrasad/LCD-OCR/blob/master/code.py

    dilate = cv2.dilate(edged, None, iterations=4)
    cv2.imwrite("dilate.jpg", dilate)

    erode = cv2.erode(dilate, None, iterations=4)
    cv2.imwrite("erode.jpg", erode)

    return dilate, erode


def scale_box(img, width, height):
    (h, w) = img.shape
    aspect = h / w
    scale = 0.0
    if aspect > 1.0:
        scale = height / h
    else:
        scale = width / w
    scaled = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    (h2, w2) = scaled.shape
    result = np.zeros((width, height, 3), np.uint8)
    mat = np.float32([[1, 0, ((width - w2) / 2.0)], [0, 1, 0]])
    return cv2.warpAffine(scaled, mat, (width, height), dst=result)


# load image
image = cv2.imread("./images/example.jpg")

# resize

# image = imutils.resize(image, height=500)

# gray
grayed = gray(image)
# denoizing

denoized = denoise(grayed)

# lut

contrast = lut(denoized, 60, 195)

# canny

edged = canny(contrast)

# filter noize

edged_cnts = filter_noise(edged)

# open

dilate, erode = closing(edged_cnts)

# find contuors

# mask2 = np.ones(image.shape[:2], dtype="uint8") * 255

cnts = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# sort contours

sorted_cnts = contours.sort_contours(cnts, method="left-to-right")[0]

# get pairs

pairs = itertools.combinations(sorted_cnts, 2)

# word group
words = []
for p in pairs:
    p0 = p[0]
    p1 = p[1]
    (x0, y0, w0, h0) = cv2.boundingRect(p0)
    (x1, y1, w1, h1) = cv2.boundingRect(p1)
    d = np.linalg.norm(np.array([x1, y1]) - np.array([x0, y0]))
    if d < h0:
        print(f"x0={x0}, y0={y0}, x1={x1}, y1={y1}, d={d}, h0={h0}")
        flat = itertools.chain.from_iterable(words)
        if len(words) <= 0:
            words.append(list([p0, p1]))
        elif p0 not in flat and p1 not in flat:
            words.append(list([p0, p1]))
        else:
            for w in words:
                if p0 in w and p1 not in w:
                    w = w.append(p1)
                elif p1 in w and p0 not in w:
                    w = w.append(p0)

print(len(words))

# read trained keras model

f = open("./train.json")
json_text = f.read()
model = model_from_json(json_text)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])
model.load_weights("./train.hdf5")

# print words
textbox = image.copy()
text = []
for i, w in enumerate(words):
    ln = []
    for j, c in enumerate(w):
        (x, y, w, h) = cv2.boundingRect(c)
        roi = contrast[y:y + h, x:x + w]  # clip numeric segment
        ret, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # TODO normalize to mnist
        th = scale_box(th, 64, 64)
        cv2.imwrite(f"th{j}.jpg", th)  # DEBUG
        ln.append(th)
    scores = model.predict(ln)
    print("scores: ", scores)

cv2.imwrite("textbox.jpg", textbox)
