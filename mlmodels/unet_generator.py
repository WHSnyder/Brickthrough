import numpy as np
import keras
import random
import cv2

class UnetGenerator(keras.utils.Sequence):

    #'Generates data for Keras'
    def __init__(self, val, batch_size=6, dim=(32,32,32), n_channels=1,
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
        self.path = "/home/will/projects/legoproj/data/kpts_dset_{}/kpts/"

        self.numdict = {0:398,1:126,2:723}


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

            i1 = random.randint(0,2)
            i2 = random.randint(0,self.numdict[i1])

            imgpath = (self.path + "{}_masked.png").format(i1,i2)
            maskpath = (self.path + "{}_outline.png").format(i1,i2) 

            img = cv2.imread(imgpath,0)
            img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
            img = np.reshape(img,(256,256,1))

            mask = cv2.imread(maskpath,0)
            mask = cv2.resize(mask,(256,256),interpolation=cv2.INTER_LINEAR)
            mask = np.reshape(mask,(256,256,1))


            x.append(img)
            y.append(mask)

        x = np.array(x).astype('float32')/255.0
        y = np.array(y).astype('float32')/255.0

        return x,y
