import jax.numpy as jnp
from utils.model import dce_net
from utils.loss_functions import SpatialConsistencyLoss
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


model = dce_net(SpatialConsistencyLoss())
params = pd.read_pickle("model_params.pkl")
if __name__ == "__main__":
    img_name = input('Enter image name: ')
    img = Image.open(f"test/{img_name}")
    img.resize((256, 256))
    img = jnp.array(img)/255
    img = img[jnp.newaxis,...]
    enhance_img = model.predict(params, img)
    # new_img = Image.fromarray(enhance_img[0].astype('uint8'), 'RGB')
    plt.imsave(f"test/enhance_{img_name}", enhance_img[0])