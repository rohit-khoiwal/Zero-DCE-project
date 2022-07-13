import os
import cv2
import jax
import jax.numpy as jnp
import pandas as pd
from model import dce_net
from loss_functions import SpatialConsistencyLoss
from utils import load_dataset, predict


print("Loading Model..............")
model = dce_net(SpatialConsistencyLoss())
params = pd.read_pickle("dce_net_params.pkl")


def enhance_image(img, size, verbose=False):
    if verbose:
        print("Enhancing the Image..........")
    enhanced_imgs = predict(model, params, img[jnp.newaxis,:])
    if verbose:
        print("Resizing the Image...........")
    enhanced_img = cv2.resize((enhanced_imgs[0]*255).astype(jnp.uint8).to_py(), (size[0], size[1]))
    return enhanced_img


if __name__=="__main__":
    img_name = input("Enter image name with extension: ")
    img_path = "tests/" + img_name

    try:
        img = cv2.imread(img_path)
        real_size = img.shape
        real_size = (real_size[1], real_size[0])
        img = cv2.resize(img, (256,256))
        img = jnp.array(img)/255
        print("Image is succesfully loaded.")
    except Exception as e:
        print("Image is not able to loaded. Please try again.")
        exit()
    
    enhanced_img = enhance_image(img, real_size, True)
    print("Saving the Image.............")
    cv2.imwrite(f"tests/enhanced_{img_name}", enhanced_img)

    