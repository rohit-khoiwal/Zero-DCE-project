import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from loss_functions import SpatialConsistencyLoss


def load(img_name):
    img = Image.open(img_name)
    img = img.resize((256,256))
    return jnp.array(img)/255

def load_dataset(names):
    return jnp.array(list(map(load, names)))

def get_enhanced_image(org_img, output):
        for i in range(0, 3 * 8, 3):
            r = output[:, :, :, i: i + 3]
            org_img = org_img + r * (jnp.square(org_img) - org_img)
        return org_img


def fit(model,params, X, batch_size=32, learning_rate=0.01, epochs=10, rng=jax.random.PRNGKey(0)):
    opt = optax.adam(learning_rate=learning_rate)
    opt_state = opt.init(params)

    loss_grad_fn = jax.value_and_grad(model.loss_fn)
    losses = []
    total_epochs = (len(X) // batch_size) * epochs

    carry = {}
    carry["params"] = params
    carry["state"] = opt_state
    @jax.jit
    def one_epoch(carry, rng):
        params = carry["params"]
        opt_state = carry["state"]
        idx = jax.random.choice(
            rng, jnp.arange(len(X)), shape=(batch_size,), replace=False
        )
        loss_val, grads = loss_grad_fn(params, X[idx])
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        carry["params"] = params
        carry["state"] = opt_state

        return carry, loss_val

    carry, losses = jax.lax.scan(one_epoch, carry, jax.random.split(rng, total_epochs))
    return carry["params"], losses


def plot_results(images, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()

def predict(model, params, imgs):
    outputs = model.apply(params, imgs)
    enhanced_imgs = get_enhanced_image(imgs,outputs)
    return enhanced_imgs