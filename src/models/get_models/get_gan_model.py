
"""
Partially adopted from https://keras.io/examples/generative/wgan_gp/
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras import layers
from matplotlib import pyplot as plt

def get_G(latent_dim, num_hid_layers, window):
    
    generator = keras.Sequential(name="generator")
    generator.add(keras.Input(shape=(latent_dim,)))

    generator.add(layers.Dense(2 * 2 * latent_dim),)
    generator.add(layers.Reshape((2, 2, latent_dim)))

    generator.add(layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding="same"))
    generator.add(layers.LeakyReLU(alpha=0.2))

    generator.add(layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding="same"))
    generator.add(layers.LeakyReLU(alpha=0.2))

    generator.add(layers.Conv2DTranspose(256, kernel_size=2, strides=(2,2), padding="same"))
    generator.add(layers.LeakyReLU(alpha=0.2))

    for _ in range(num_hid_layers):
        generator.add(layers.Conv2DTranspose(256, kernel_size=2, strides=(1,1), padding="same"))
        generator.add(layers.LeakyReLU(alpha=0.2))

    # Last Layer...
    generator.add(layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding="same",  activation="linear"))
    if window == 6:
        generator.add(layers.Cropping2D((5, 2)))
    elif window == 8:
        generator.add(layers.Cropping2D((4, 2)))
    elif window == 10:
        generator.add(layers.Cropping2D((3, 2)))
    elif window == 12:
        generator.add(layers.Cropping2D((2, 2)))
    generator.summary()
    return generator


def get_D(time_step, no_feat, num_hid_layers):
# --------------------------------
    discriminator = keras.Sequential(name="discriminator")
    discriminator.add(keras.Input(shape=(time_step, no_feat, 1)))

    discriminator.add(layers.Conv2D(128, kernel_size=2, strides=2, padding="same"))
    discriminator.add(layers.LeakyReLU(alpha=0.2))

    discriminator.add(layers.Conv2D(256, kernel_size=2, strides=2, padding="same"))
    discriminator.add(layers.LeakyReLU(alpha=0.2))

    discriminator.add(layers.Conv2D(256, kernel_size=2, strides=2, padding="same"))
    discriminator.add(layers.LeakyReLU(alpha=0.2))

    for _ in range(num_hid_layers):
        discriminator.add(layers.Conv2D(256, kernel_size=2, strides=1, padding="same"))
        discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dropout(0.2))
    discriminator.add(layers.Dense(1, activation="linear"))

    discriminator.summary()

    return discriminator


class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss} #TODO: Update to average loss



## Create a callback that periodically saves generated images
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        if epoch%25 ==0:
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            generated_images = self.model.generator(random_latent_vectors)
            generated_images = (generated_images * 127.5) + 127.5

def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)



def get_wgan(cfg, model_cfg, models_dict):

    window = cfg.window
    num_signals = len(cfg.features)
    noise_dim = model_cfg.noise_dim
    num_hid_layers = model_cfg.num_hid_layers
    learning_rate_wgan = cfg.learning_rate_wgan
    
    if cfg.device == 'gpu':
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            try:
                g_model = keras.models.load_model(models_dict["generator"])
                d_model = keras.models.load_model(models_dict["discriminator"])
                print(f"Loaded trained model")
            except:
                g_model = get_G(noise_dim, num_hid_layers, window)
                d_model = get_D(window, num_signals, num_hid_layers)
                print("Creating new model")

            generator_optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate_wgan, beta_1=0.5, beta_2=0.9
            )
            discriminator_optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate_wgan, beta_1=0.5, beta_2=0.9
            )

            # Instantiate the customer `GANMonitor` Keras callback.
            cbk = GANMonitor(num_img=3, latent_dim=noise_dim)
                
            # Get the wgan model
            wgan = WGAN(
                discriminator=d_model,
                generator=g_model,
                latent_dim=noise_dim,
                discriminator_extra_steps=3,
            )
            # Compile the wgan model
            wgan.compile(
                d_optimizer=discriminator_optimizer,
                g_optimizer=generator_optimizer,
                g_loss_fn=generator_loss,
                d_loss_fn=discriminator_loss,
            )

            print("Model compiled")
    else:
        
        try:
            g_model = keras.models.load_model(models_dict["generator"])
            d_model = keras.models.load_model(models_dict["discriminator"])
            print(f"Loaded trained model")
        except:
            g_model = get_G(noise_dim, num_hid_layers, window)
            d_model = get_D(window, num_signals, num_hid_layers)
            print("Creating new model")

        generator_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate_wgan, beta_1=0.5, beta_2=0.9
        )
        discriminator_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate_wgan, beta_1=0.5, beta_2=0.9
        )

        # Instantiate the customer `GANMonitor` Keras callback.
        cbk = GANMonitor(num_img=3, latent_dim=noise_dim)
            
        # Get the wgan model
        wgan = WGAN(
            discriminator=d_model,
            generator=g_model,
            latent_dim=noise_dim,
            discriminator_extra_steps=3,
        )
        # Compile the wgan model
        wgan.compile(
            d_optimizer=discriminator_optimizer,
            g_optimizer=generator_optimizer,
            g_loss_fn=generator_loss,
            d_loss_fn=discriminator_loss,
        )

        print("Model compiled")

    return wgan, cbk