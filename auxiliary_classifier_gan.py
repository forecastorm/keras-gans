from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
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

        model.summary()









if __name__ == '__main__':
    auxiliary_classifier_gan = ACGAN()
