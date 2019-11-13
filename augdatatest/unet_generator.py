import numpy as np
import keras
import random

class UnetGenerator(keras.utils.Sequence):

    #'Generates data for Keras'
    def __init__(self, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        #'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path = "/home/will/projects/legoproj/data/normalz/"

        random.seed(0)


    def __len__(self):
#        'Denotes the number of batches per epoch'
        return 20


    def __getitem__(self,index):
        #'Generate one batch of data'
        return self.__data_generation()


    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = 0


    def __data_generation(self):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x=[]
        y=[]

        # Generate data
        for i in range(self.batch_size):

        	i1 = random.randint(0,1)
        	i2 = random.randint(0,2000)

        	imgpath = self.path + "{}_{}.png".format(i1,i2)
        	normalspath = self.path + "{}_{}_normz.png".format(i1,i2) 

            img = cv2.imread(imgpath,0)
            img = np.reshape(img,(512,512,1))
            normals = cv2.imread(normalspath)

            x.append(img)
            y.append(normals)

        x = np.array(x).astype('float32')/255.0
        y = np.array(y).astype('float32')/255.0

        return x,y
