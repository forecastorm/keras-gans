from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.utils.vis_utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import random


def show_model(model):
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# read in gray scale
def load_image(file_path):
    return cv2.imread(file_path, 0)


def extract_label(file_name):
    return 1 if "open" in file_name else 0  # open eyes are 1 & closed eyes are 0


class ACGAN():
    def __init__(self):
        # input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100
        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])
        # build the generator
        self.generator = self.build_generator()

        # pass noise and target label to the generator label
        # Tensor with shape(?, 100)
        # So the latent space is the space for noise?
        noise = Input(shape=(self.latent_dim,))
        # Tensor with shape(?,1)
        label = Input(shape=(1,))
        # Tensor with shape(?,14,14,1)
        img = self.generator([noise, label])

        # going to combine generator and discriminator as generator
        # the discriminator compiled before, is trained
        # the combined model, to use for generate, is not trainable anymore
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        # This model concludes generator, then discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses, optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        # number of params = output_size * (input_size + 1)
        # output shape(None, 6272) , params (633472)
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        # output shape(None, 7, 7, 128), params (0)
        model.add(Reshape((7, 7, 128)))
        # output shape (None,7, 7, 128), params (512)
        model.add(BatchNormalization(momentum=0.8))
        # up sample the shape to be (None, 14, 14, 128)
        model.add(UpSampling2D())
        # output shape(None,14,14,64), params (73792)
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))
        # A tensor with shape (?, 100)
        noise = Input(shape=(self.latent_dim,))
        # A tensor with shape (?, 1)
        label = Input(shape=(1,), dtype='int32')
        # A tensor with shape ( ? ,?)
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)
        return Model([noise, label], img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        # Tensor with shape (?,28,28,1)
        img = Input(shape=self.img_shape)
        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        # Tensor with shape (?,1)
        validity = Dense(1, activation="sigmoid")(features)
        # Tensor with shape (?,10)
        label = Dense(self.num_classes, activation="softmax")(features)
        return Model(img, [validity, label])

    # introduce sampling interval to produce random selection
    def train(self, epochs, batch_size=32, sample_interval=50):
        # who does not like mnist
        # already numpy array
        # X_train shape by (6000,28,28)
        # y_train shape by (6000,)
        # (X_train, y_train), (_, _) = mnist.load_data()
        '''substitute to train eye images '''
        img_path = "./BW28x28/"
        img_files = []
        # there was a ghost file getting added to this
        for file in os.listdir(img_path):
            if file.endswith(".png"):
                img_files.append(file)

        random.shuffle(img_files)
        # load all images from folder, split them later
        X_train = np.asarray([load_image(img_path + file) for file in img_files])
        y_train = np.asarray ([extract_label(file) for file in img_files])
        '''substitute to train eye images '''

        # data pre-manipulation
        # normalize to float between -1 and 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # same as expand dim axis = -1
        # X_train shape by (6000,28,28,1)
        X_train = np.expand_dims(X_train, axis=3)
        # shape by (6000,1)
        y_train = y_train.reshape(-1, 1)

        # adversarial ground truths
        # valid shape by (32,1)
        # fake shape by (32,1)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # select a random batch of images
            # random batch_size ints that is between 0 to 60000
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs shape by (32, 28, 28, 1)
            imgs = X_train[idx]

            # sample noise as generator input
            # noise shape by (32, 100), mean 0, standard deviation 1
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The labels of the digits that generator tries to create
            # initialized to random, let it blind guess first
            # shaped by (32,1)
            sampled_labels = np.random.randint(0, 10, (batch_size, 1))
            # generate a batch of the new images
            # gen_imgs shape by (32, 14, 14, 1)
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # the legit image labels
            # img_labels shape by (32, 1)
            img_labels = y_train[idx]

            # train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train the combined generator
            # generator loss
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
            epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

            # if at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_image(epoch)

    def sample_image(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):
        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    auxiliary_classifier_gan = ACGAN()
    # training 14000 times,
    # each time 32 samples
    # save every generated sample 200 interval from the next
    auxiliary_classifier_gan.train(epochs=14000, batch_size=16, sample_interval=200)
