from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.utils.vis_utils import plot_model
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
        self.build_generator()


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
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)
        print(noise)
        # model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
if __name__ == '__main__':
    auxiliary_classifier_gan = ACGAN()

