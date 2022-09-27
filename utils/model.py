import flax.linen as nn
import jax.numpy as jnp
from utils.loss_functions import *
from utils.utils import get_enhanced_image


class dce_net(nn.Module):
    loss_spa: None

    @nn.compact
    def __call__(self, x):

        x1 = nn.Conv(32, kernel_size=(3, 3))(x)
        x1 = nn.relu(x1)
        x2 = nn.Conv(32, kernel_size=(3, 3))(x1)
        x2 = nn.relu(x2)
        x3 = nn.Conv(32, kernel_size=(3, 3))(x2)
        x3 = nn.relu(x3)
        x4 = nn.Conv(32, kernel_size=(3, 3))(x3)
        x4 = nn.relu(x4)

        new_x1 = jnp.concatenate([x4, x3], axis=-1)
        x5 = nn.Conv(32, kernel_size=(3, 3))(new_x1)
        x5 = nn.relu(x5)

        new_x2 = jnp.concatenate([x5, x2], axis=-1)
        x6 = nn.Conv(32, kernel_size=(3, 3))(new_x2)
        x6 = nn.relu(x6)

        new_x3 = jnp.concatenate([x6, x1], axis=-1)
        x7 = nn.Conv(24, kernel_size=(3, 3))(new_x3)
        output = nn.tanh(x7)
        return output

    def loss_fn(self, params, org):
        output = self.apply(params, org)
        enhanced_y = get_enhanced_image(org, output)

        loss_ls = 200 * illumination_smoothness_loss(output)
        loss_spa = jnp.mean(self.loss_spa(org, enhanced_y))
        loss_color = 5 * jnp.mean(color_constancy_loss(enhanced_y))
        loss_exp = 10 * jnp.mean(exposure_loss(enhanced_y))

        total_loss = loss_ls + loss_spa + loss_color + loss_exp

        return total_loss

    def predict(self, params, x):
        output = self.apply(params, x)
        return get_enhanced_image(x, output)
