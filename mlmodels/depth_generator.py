import numpy as np
import keras
import random
import cv2

class DepthGenerator(keras.utils.Sequence):

    #'Generates data for Keras'
    def __init__(self, val, batch_size=8, dim=(32,32,32), n_channels=1,
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
        self.path = "/home/will/projects/legoproj/data/kpts_dset_{}/"

        self.numdict = {0:499,1:299,2:499}#,1:410,2:1204}


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

            #i1 = random.randint(0,2)
            #i2 = random.randint(0,self.numdict[i1])

            i1 = random.randint(0,6)

            if self.val:
                i2 = random.randint(180,199)
            else:
                i2 = random.randint(0,180)

            tag = "{:0>4}".format(i2)

            imgpath = (self.path + "{}_a.png").format(i1,tag)
            depthpath = (self.path + "{}_npdepth.npy").format(i1,tag) 
            maskpath = (self.path + "{}_masks.png").format(i1,tag)

            img = cv2.imread(imgpath,0)
            img = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
            img = np.reshape(img,(256,256,1))

            mask = cv2.resize(cv2.imread(maskpath,0), (256,256), interpolation=cv2.INTER_LINEAR)
            mask = np.reshape(mask,(256,256,1)).astype(np.float32)

            d = np.load(depthpath,allow_pickle=False)
            d = np.reshape(d,(512,512,1)).astype(np.float32)
            depth = d[0::2,0::2]
            depth[depth > 9999.0] = 0.0

            depth = np.concatenate((depth,mask),axis=-1)
            
            x.append(img)
            y.append(depth)

        x = np.array(x).astype('float32')/255.0
        y = np.array(y).astype('float32')

        return x,y
