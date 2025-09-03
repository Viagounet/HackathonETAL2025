import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

##exemple de path pour les images et de masque##
path = "image.png"
path2 = "image_blurred.png"

mask = np.zeros((841, 861, 3), dtype=np.uint8)
mask = cv2.circle(mask, center=(400, 400), radius=100, color=(255,255,255), thickness=-1)
####

def flou(path, n):
    img = cv2.imread(path)
    blurred_img = cv2.GaussianBlur(img, (n, n), 0)
    cv2.imwrite(path.split('.')[0]+"_blurred.png", blurred_img)

def enleve_masque(img_unblurred, img_blurred, mask): #img_blurred correspond à la dernière image affiché dans le chat
    img = cv2.imread(img_unblurred)
    blurred_img = cv2.imread(img_blurred)
    out = np.where(mask==np.array([255, 255, 255]), img, blurred_img)
    cv2.imwrite(img_blurred, out) #je remplace directement img_blurred par la nouvelle version

##tests##
flou(path, 61)
enleve_masque(path, path2, mask)