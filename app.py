import jax.numpy as jnp
from model  import dce_net
from loss_functions import SpatialConsistencyLoss
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


model = dce_net(SpatialConsistencyLoss())
params = pd.read_pickle("model_params.pkl")
if __name__ == "__main__":
    img_name = input('Enter Image name with directory : ')
    img = Image.open(img_name)
    img.resize((256, 256))
    img = jnp.array(img)/255
    img = img[jnp.newaxis,...]
    enhance_img = model.predict(params, img)
    plt.imsave('test.png', enhance_img[0])