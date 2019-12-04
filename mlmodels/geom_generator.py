import numpy as np
import keras
import random
import cv2

class Geom_Generator(keras.utils.Sequence):

    #'Generates data for Keras'
    def __init__(self, val, batch_size=5, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        #'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = []
        self.list_IDs = []
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.val = val
        self.on_epoch_end()
        self.path = "/home/will/projects/legoproj/data/{}_geom/"

        self.mapping = {0:198,1:519,2:251,3:477,4:237,5:236,6:38}

        random.seed(0)


    def __len__(self):
        #'Denotes the number of batches per epoch'
        if self.val:
            return 10
        return 100


    def __getitem__(self,index):
        #'Generate one batch of data'
        return self.__data_generation()


    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = 0


    def __data_generation(self):
        
        x=[]
        y=[]

        # Generate data
        for i in range(self.batch_size):

            i1 = random.randint(0,6)
            i2 = random.randint(0,self.mapping[i1]-1)

            imgpath = (self.path + "{}.png").format(i1,i2)
            maskpath = (self.path + "{}_geom.png").format(i1,i2) 

            img = cv2.imread(imgpath,0)
            img = cv2.resize(img,(128,128),interpolation=cv2.INTER_LINEAR)
            img = np.reshape(img,(128,128,1))

            mask = cv2.imread(maskpath)
            mask = cv2.resize(mask,(128,128),interpolation=cv2.INTER_LINEAR)
            mask = np.reshape(mask,(128,128,3))

            x.append(img)
            y.append(mask)


        x = np.array(x).astype('float32')/255.0
        y = np.array(y).astype('float32')/255.0


        return x,y
