import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

##exemple de path pour les images et de masque##
path = "image.png"
path2 = "image_blurred.png"

mask = np.zeros((841, 861, 3), dtype=np.uint8)
mask = cv2.circle(mask, center=(400, 400), radius=100, color=(255,255,255), thickness=-1)

mask1 = np.zeros((4, 4, 3), dtype=np.uint8)
mask2 = cv2.circle(copy.deepcopy(mask1), center=(2, 2), radius=2, color=(255,255,255), thickness=-1)
####

def flou(pil, n):
    img = pipo(pil)
    blurred_img = cv2.GaussianBlur(img, (n, n), 0)
    cv2.imwrite("blurred.png", blurred_img)
    return blurred_img

def enleve_masque(img_unblurred, img_blurred, prev_mask, new_mask): #img_blurred correspond à la dernière image affiché dans le chat
    img = pipo(img_unblurred)
    blurred_img = pipo(img_blurred)
    mask = fusion_masque(prev_mask, new_mask)
    out = np.where(mask==np.array([255,255,255]), img, blurred_img)
    return out, mask

def fusion_masque(prev_mask, new_mask):
    n,m,l = np.shape(prev_mask)
    for i in range(n):
        for j in range(m):
            if np.all(new_mask[i,j]):
                prev_mask[i,j] = [255,255,255]
    return(prev_mask)

def pourcentage_reussite(mask):
    c = 0
    n,m,l = np.shape(mask)
    for i in range(n):
        for j in range(m):
            if np.all(mask[i,j]):
                c += 1
    percent = (c/(n*m))*100
    return percent


def pipo(pil_image):
    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    return opencvImage

##tests##
#enleve_masque(path, path2, mask)
print(pourcentage_reussite(mask2))
