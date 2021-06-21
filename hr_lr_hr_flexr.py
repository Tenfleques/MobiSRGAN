# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import time
from random import randint
from matplotlib import pyplot as plt
import cv2


AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_ROOT = "data/FlexR/"
WORK_DIR = "lrgan/"
DIV2K_ROOT = "data/DIV2K/"
WEIGHTS_DIR = "weights"
TRAIN_LR_IMGS = "256x144"
TRAIN_HR_IMGS = "1920x1080"

SIZE_X = [1920, 1080]
SIZE_Y = [256, 144]

BATCH_SIZE = 32
BUFFER_SIZE = 400
IMG_HEIGHT = 96
IMG_WIDTH = 96

os.makedirs(WORK_DIR, exist_ok=True)
get_name = lambda x : ("0"*5 + str(x))[-5:]

def decode_img(img, size):
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)

    return img 

def open_file(x, y):
    img_x = tf.io.read_file(x)
    img_x = decode_img(img_x, SIZE_X)

    img_y = tf.io.read_file(y)
    img_y = decode_img(img_y, SIZE_Y)

    return img_x, img_y

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

def resize(input_image, real_image, height = IMG_HEIGHT, width = IMG_WIDTH):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def clean_images(img_dir):
  images = os.listdir(img_dir)
  hrs = [hr for hr in images if "{}x{}".format(*SIZE_X) in hr]

  for i in hrs:
    lr_counter_part = os.path.join(img_dir, i.replace("{}x{}".format(*SIZE_X), "{}x{}".format(*SIZE_Y)))
    if not os.path.exists(lr_counter_part):
      fp = os.path.join(img_dir, i)
      os.remove(fp)
      print("[INFO] removed {}".format(fp))

clean_images(IMG_ROOT)

def load_flexr_4k_eager(index, hr, lr, img_dir):
    hr_img = "{}{}-{}.png".format(img_dir, get_name(index), hr)
    lr_img = "{}{}-{}.png".format(img_dir, get_name(index), lr)
    if not os.path.exists(hr_img):
        while True:
            hr_img = "{}{}-{}.png".format(img_dir, get_name(index), hr)
            lr_img = "{}{}-{}.png".format(img_dir, get_name(index), lr)

            if os.path.exists(hr_img):
                break

    hr_img, lr_img = open_file(hr_img, lr_img)

    hr_img, lr_img = resize(hr_img, lr_img)
    return hr_img, lr_img

def check_flexr():
    hr, lr = load_flexr_4k_eager(randint(0,100), TRAIN_HR_IMGS, TRAIN_LR_IMGS, IMG_ROOT)
    instance = randint(0, 1000)

    lr = tf.keras.preprocessing.image.img_to_array(lr)
    hr = tf.keras.preprocessing.image.img_to_array(hr)

    cv2.imwrite(os.path.join(WORK_DIR, "preview", "lr-preview-{}.png".format(instance)), lr)
    cv2.imwrite(os.path.join(WORK_DIR, "preview", "hr-preview-{}.png".format(instance)), hr)

check_flexr()

@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

hr, lr = load_flexr_4k_eager(randint(0,100), TRAIN_HR_IMGS, TRAIN_LR_IMGS, IMG_ROOT)
rj_inp, rj_re = random_jitter(hr, lr)

def visulaize_random_data():
    instance = randint(0, 1000)
    hr, lr = load_flexr_4k_eager(randint(0,100), TRAIN_HR_IMGS, TRAIN_LR_IMGS, IMG_ROOT)
    rj_inp, rj_re = random_jitter(hr, lr)

    rj_inp = tf.keras.preprocessing.image.img_to_array(rj_inp)
    rj_re = tf.keras.preprocessing.image.img_to_array(rj_re)

    cv2.imwrite(os.path.join(WORK_DIR, "preview", "lr-random-jitter-{}.png".format(instance)), rj_re)


    cv2.imwrite(os.path.join(WORK_DIR, "preview", "hr-random-jitter-{}.png".format(instance)), rj_inp)

visulaize_random_data()

"""## Input Pipeline"""

def load_flexr_4k_image_tf(hr):
    parts = tf.strings.split(hr, os.path.sep)
    needle = tf.strings.split(parts[-1], '-')[-1]
    lr = tf.strings.regex_replace(hr, needle, "{}.png".format(TRAIN_LR_IMGS))
    
    hr_img, lr_img = open_file(hr, lr)
    hr_img, lr_img = resize(hr_img, lr_img,
                                    IMG_HEIGHT, IMG_WIDTH)
    hr_img, lr_img = normalize(hr_img, lr_img)
    return hr_img, lr_img

dataset = tf.data.Dataset.list_files(IMG_ROOT + '*-{}.png'.format(TRAIN_HR_IMGS))
dataset = dataset.map(load_flexr_4k_image_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)

DATASET_SIZE = 0
for a in dataset:
  DATASET_SIZE += 1

train_size = int(0.7 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

full_dataset = dataset.shuffle(BATCH_SIZE)

train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)

val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(rj_inp, 0))
print (down_result.shape)

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)

def Generator():
  inputs = tf.keras.layers.Input(shape=[None,None,3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

gen_output = generator(rj_inp[tf.newaxis,...], training=False)
cv2.imwrite(os.path.join(WORK_DIR, "preview", "gen-output.png"), gen_output[0,...])

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

disc_out = discriminator([rj_inp[tf.newaxis,...], gen_output], training=False)
cv2.imwrite(os.path.join(WORK_DIR, "preview", "gen-output.png"), disc_out[0,...,-1] * 255)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = '{}training_checkpoints'.format(WORK_DIR)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

import datetime
log_dir="{}logs/".format(WORK_DIR)

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def generate_images(model, test_input, tar, epoch):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  
  plt.savefig(os.path.join(WORK_DIR, "train_step-{}.png".format(epoch)))

@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs, test_ds, start=0):
  for epoch in range(start, epochs):
    start = time.time()
    
    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target, epoch)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)


EPOCHS = 250
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


fit(train_dataset, EPOCHS, test_dataset, start=200)

generator.save(os.path.join(WEIGHTS_DIR, "lr_generator_2.h5"))