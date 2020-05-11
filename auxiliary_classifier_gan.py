from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.utils.vis_utils import plot_model
from keras.layers.advanced_activations import LeakyReLU

def show_model(model):
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


class ACGAN():
    def __init__(self):
        #input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.num_classes = 10
        self.latent_dim = 100
        optimizer = Adam(0.0002,0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
        # build and compile the discriminator
        self.generator=self.build_generator()
        self.build_discriminator()


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
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        show_model(model)
        # Tensor with shape (?,28,28,1)
        img = Input(shape=self.img_shape)
        print(img)
        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        # Tensor with shape (?,1)
        validity = Dense(1, activation="sigmoid")(features)
        # Tensor with shape (?,10)
        label = Dense(self.num_classes, activation="softmax")(features)
        return Model(img,[validity,label])



if __name__ == '__main__':
    auxiliary_classifier_gan = ACGAN()





