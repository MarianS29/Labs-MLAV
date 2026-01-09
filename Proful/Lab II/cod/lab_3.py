import numpy as np
# Nu trebuie tf pentru citirea datelor, dar trebuie pentru tot restul
import torch 
def get_MNIST_train():
    
    mnist_train_data = np.zeros([60000,784])
    mnist_train_labels = np.zeros(60000)
    
    f = open('train-images.idx3-ubyte','r',encoding = 'latin-1')
    g = open('train-labels.idx1-ubyte','r',encoding = 'latin-1')
    
    byte = f.read(16) #4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
    byte_label = g.read(8) #4bytes magic number, 4 bytes nr labels
    
    mnist_train_data = np.fromfile(f,dtype=np.uint8).reshape(60000,784)
    mnist_train_data = mnist_train_data.reshape(60000,1,28,28)
    mnist_train_labels = np.fromfile(g,dtype=np.uint8)
    
    # Conversii pentru a se potrivi cu procesul de antrenare    
    mnist_train_data = mnist_train_data.astype(np.float32)
    mnist_train_labels = mnist_train_labels.astype(np.int64)
    
    return mnist_train_data, mnist_train_labels

def get_MNIST_test():
    
    mnist_test_data = np.zeros([10000,784])
    mnist_test_labels = np.zeros(10000)
    
    f = open('t10k-images.idx3-ubyte','r',encoding = 'latin-1')
    g = open('t10k-labels.idx1-ubyte','r',encoding = 'latin-1')
    
    byte = f.read(16) #4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
    byte_label = g.read(8) #4bytes magic number, 4 bytes nr labels
    
    mnist_test_data = np.fromfile(f,dtype=np.uint8).reshape(10000,784)
    mnist_test_data = mnist_test_data.reshape(10000,1,28,28)
    mnist_test_labels = np.fromfile(g,dtype=np.uint8)
        
    # Conversii pentru a se potrivi cu procesul de antrenare    
    mnist_test_data = mnist_test_data.astype(np.float32)
    mnist_test_labels = mnist_test_labels.astype(np.int64)
    
    return mnist_test_data, mnist_test_labels
