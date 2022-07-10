from curses.ascii import SP
import os

import cv2
import jax.numpy as jnp
import pandas as pd
from model import dce_net
from loss_functions import SpatialConsistencyLoss
from utils import load_dataset, predict, plot_results
from PIL import Image
import matplotlib.pyplot as plt

img_name = input("Enter image name with extension: ")
img_path = "test_imgs/" + img_name

try:
    img = cv2.imread(img_path)
    real_shape = img.shape
    img = cv2.resize(img, (256,256))
    img = jnp.array(img)/255
    imgs = jnp.array([img])
    print("Image is succesfully loaded.")
except Exception as e:
    print("Image is not able to loaded. Please try again.")
    exit()

print("Loading Model..............")
model = dce_net(SpatialConsistencyLoss())
params = pd.read_pickle("dce_net_params.pkl")

print("Enhancing the Image........")
enhanced_imgs = predict(model, params, imgs)
print("Saving the Image...........")
enhanced_img = cv2.resize(enhanced_imgs[0].to_py()*255, (real_shape[1], real_shape[0]))
print(enhanced_img.shape)
cv2.imwrite(f"test_imgs/enhanced_{img_name}", enhanced_img)


# plt.ion()
# plot_results([imgs[0], enhanced_imgs[0]])
# plt.show()


# vidcap = cv2.VideoCapture('VID.mp4')
# success, image = vidcap.read()
# count = 0
# while success:
#   cv2.imwrite(f"new_imgs/frame{count}.jpg", image)     # save frame as JPEG file      
#   success, image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1

# print(count)