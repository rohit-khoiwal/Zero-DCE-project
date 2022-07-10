import jax
import jax.numpy as jnp
from functools import partial
import flax.linen as nn

def color_constancy_loss(x):
    mean_rgb = jnp.mean(x, axis=(1, 2), keepdims=True)
    mean_red = mean_rgb[:, :, :, 0]
    mean_green = mean_rgb[:, :, :, 1]
    mean_blue = mean_rgb[:, :, :, 2]
    diff_red_green = jnp.square(mean_red - mean_green)
    diff_red_blue = jnp.square(mean_red - mean_blue)
    diff_green_blue = jnp.square(mean_blue - mean_green)
    return jnp.sqrt(
        jnp.square(diff_red_green) + jnp.square(diff_red_blue) + jnp.square(diff_green_blue)
    )

def exposure_loss(x, mean_val=0.6):
    x = jnp.mean(x, axis=3, keepdims=True)
    mean = nn.avg_pool(x, (16,16), strides=(16,16))
    return jnp.mean(jnp.square(mean - mean_val))

def illumination_smoothness_loss(x):
    batch_size = x.shape[0]
    height_x = x.shape[1]
    width_x = x.shape[2]
    
    count_height = (width_x - 1)*x.shape[3]
    count_width = width_x*(x.shape[3] - 1)
    
    heigth_total_variance = jnp.sum(jnp.square(x[:, 1:, :, :] - x[:, :height_x - 1, :, :]))
    width_total_variance = jnp.sum(jnp.square(x[:, :, 1:, :] - x[:, :, :width_x - 1, :]))
    
    return 2*(heigth_total_variance/count_height  + width_total_variance/count_width)/batch_size



class SpatialConsistencyLoss():
    def __init__(self):
        self.conv = partial(nn.Conv, features=1, kernel_size=(3,3))
        self.left_conv = self.conv(kernel_init=self.init_lk)
        self.right_conv = self.conv(kernel_init=self.init_rk)
        self.up_conv = self.conv(kernel_init=self.init_uk)
        self.down_conv = self.conv(kernel_init=self.init_dk)
        
        self.left_params = self.init_params(self.left_conv, jnp.ones([1, 64, 64, 1]))
        self.right_params = self.init_params(self.right_conv, jnp.ones([1, 64, 64, 1]))
        self.up_params = self.init_params(self.up_conv, jnp.ones([1, 64, 64, 1]))
        self.down_params = self.init_params(self.right_conv, jnp.ones([1, 64, 64, 1]))
        
    def init_lk(self, *args):
        return  jnp.array([
            [[[0]], [[0]], [[0]]], [[[-1]], [[1]], [[0]]], [[[0]], [[0]], [[0]]]
        ])
    
    def init_rk(self, *args):
        return jnp.array([
            [[[0]], [[0]], [[0]]], [[[0]], [[1]], [[-1]]], [[[0]], [[0]], [[0]]]
        ])
    
    def init_uk(self, *args):
        return jnp.array([
            [[[0]], [[-1]], [[0]]], [[[0]], [[1]], [[0]]], [[[0]], [[0]], [[0]]]
        ])
    
    def init_dk(self, *args):
        return jnp.array([
            [[[0]], [[0]], [[0]]], [[[0]], [[1]], [[0]]], [[[0]], [[-1]], [[0]]]
        ])
    
    def init_params(self, conv, x):
        return conv.init(jax.random.PRNGKey(0), x)
    
    def __call__(self, org, enhance):
        org_mean = jnp.mean(org,axis=3,keepdims=True)
        enhance_mean = jnp.mean(enhance,axis=3,keepdims=True)
        
        org_pool = nn.avg_pool(org_mean, window_shape=(4,4), strides=(4,4))
        enhance_pool = nn.avg_pool(enhance_mean, window_shape=(4,4), strides=(4,4))      
        
        
        # left kernel
        d_org_left = self.left_conv.apply(self.left_params, org_pool)
        d_enhance_left = self.left_conv.apply(self.left_params, enhance_pool)
        
        #right kernel
        d_org_right = self.right_conv.apply(self.right_params, org_pool)
        d_enhance_right = self.right_conv.apply(self.right_params, enhance_pool)
        
        #up kernel
        d_org_up = self.up_conv.apply(self.up_params, org_pool)
        d_enhance_up = self.up_conv.apply(self.up_params, enhance_pool)
        
        #down kernel
        d_org_down = self.down_conv.apply(self.down_params, org_pool)
        d_enhance_down = self.down_conv.apply(self.down_params, enhance_pool)
        
        d_left = jnp.square(d_org_left - d_enhance_left)
        d_right = jnp.square(d_org_right - d_enhance_right)
        d_up = jnp.square(d_org_up - d_enhance_up)
        d_down = jnp.square(d_org_down - d_enhance_down)
        return d_left + d_right + d_up + d_down