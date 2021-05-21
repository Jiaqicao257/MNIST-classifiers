"""
@authors: Jian Kang
@date: 11/15/2020
"""
import gzip
import numpy as np


class DataLoader(object):
    """
    class to load MNIST data
    """
    def __init__(self, Xtrainpath, Ytrainpath, Xtestpath, Ytestpath):
        self.Xtrainpath = Xtrainpath
        self.Ytrainpath = Ytrainpath
        self.Xtestpath = Xtestpath
        self.Ytestpath = Ytestpath

    @staticmethod
    def get_images(f):
        with gzip.GzipFile(fileobj=f) as bytefile:
            magic = np.frombuffer(bytefile.read(4), dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
            if magic != 2051:
                raise ValueError('Invalid magic number {} for file {}.'.format(magic, f.name))
            params = list()
            for _ in range(3):
                params.append(np.frombuffer(bytefile.read(4), dtype=np.dtype(np.uint32).newbyteorder('>'))[0])
            buffer = bytefile.read(params[0] * params[1] * params[2])
            images = np.frombuffer(buffer, dtype=np.uint8)
            return images.reshape(params[0], params[1], params[2])

    @staticmethod
    def get_labels(f):
        with gzip.GzipFile(fileobj=f) as bytefile:
            magic = np.frombuffer(bytefile.read(4), dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
            if magic != 2049:
                raise ValueError('Invalid magic number {} for file {}.'.format(magic, f.name))
            param = np.frombuffer(bytefile.read(4), dtype=np.dtype(np.uint32).newbyteorder('>'))[0]
            buffer = bytefile.read(param)
            labels = np.frombuffer(buffer, dtype=np.uint8)
            return labels

    def load_data(self):
        with open(self.Xtrainpath, 'rb') as f:
            Xtrain = self.get_images(f)
        with open(self.Ytrainpath, 'rb') as f:
            Ytrain = self.get_labels(f)
        with open(self.Xtestpath, 'rb') as f:
            Xtest = self.get_images(f)
        with open(self.Ytestpath, 'rb') as f:
            Ytest = self.get_labels(f)
        return Xtrain, Ytrain, Xtest, Ytest


dataloader = DataLoader(Xtrainpath='data/train-images-idx3-ubyte.gz',
                        Ytrainpath='data/train-labels-idx1-ubyte.gz',
                        Xtestpath='data/t10k-images-idx3-ubyte.gz',
                        Ytestpath='data/t10k-labels-idx1-ubyte.gz')
Xtrain, Ytrain, Xtest, Ytest = dataloader.load_data()
