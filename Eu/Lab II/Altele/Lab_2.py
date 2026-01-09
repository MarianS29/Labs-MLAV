import struct
import numpy as np
# Nu trebuie torch pentru citirea datelor, dar trebuie pentru tot restul
import torch

import requests
import pdb


# ~ from sh import gunzip



# ~ def unzip_file(in_file, out_file):

	# ~ gunzip(in_file)
	# ~ return True


def get_MNIST_train():
    
    mnist_train_data = np.zeros([60000,784])
    mnist_train_labels = np.zeros(60000)
    
    # ~ url = 'https://github.com/golbin/TensorFlow-MNIST/tree/master/mnist/data/train-images-idx3-ubyte.gz'
    # ~ r = requests.get(url, allow_redirects=True)
    # ~ open('train-images-idx3-ubyte.gz', 'wb').write(r.content)
    
    # ~ url = 'https://github.com/golbin/TensorFlow-MNIST/tree/master/mnist/datatrain-images-idx3-ubyte.gz'
    # ~ r = requests.get(url, allow_redirects=True)
    # ~ open('train-labels.idx1-ubyte.gz', 'wb').write(r.content)
    
    f = open('train-images.idx3-ubyte','r',encoding = 'latin-1')
    
    g = open('train-labels.idx1-ubyte','r',encoding = 'latin-1')
    
    byte = f.read(16) #4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
    byte_label = g.read(8) #4 bytes magic number, 4 bytes nr labels
    
    pdb.set_trace()
    mnist_train_data = np.fromfile(f,dtype=np.uint8).reshape(60000,784)
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
    byte_label = g.read(8) #4 bytes magic number, 4 bytes nr labels
    
    mnist_test_data = np.fromfile(f,dtype=np.uint8).reshape(10000,784)
    mnist_test_labels = np.fromfile(g,dtype=np.uint8)
    
    # Conversii pentru a se potrivi cu procesul de testare    
    mnist_test_data = mnist_test_data.astype(np.float32)
    mnist_test_labels = mnist_test_labels.astype(np.int64)        
    
    return mnist_test_data, mnist_test_labels
    
    
# Modulul nn contine o multitudine de elemente
# esentiale construirii unei retele neuronale
import torch.nn as nn

class Retea_MLP(nn.Module):
    
    def __init__(self, nr_neuroni_input, nr_neuroni_hidden, nr_clase):
        
        # Pentru a putea folosi mai departe reteaua, este recomandata mostenirea
        # clasei de baza nn.Module
        super(Retea_MLP,self).__init__()
        
        # Definirea ponderilor si a deplasamentelor din stratul ascuns
        self.w_h = torch.randn(nr_neuroni_input, nr_neuroni_hidden, dtype = torch.float, requires_grad=True)
        self.b_h = torch.randn(nr_neuroni_hidden, dtype = torch.float, requires_grad=True)
        
        # Definirea ponderilor si a deplasamentelor din stratul de iesire
        self.w_o = torch.randn(nr_neuroni_hidden, nr_clase, dtype = torch.float, requires_grad=True)
        self.b_o = torch.randn(nr_clase, dtype = torch.float, requires_grad=True)
        
    
    # Se aduna toate variabilele antrenabile intr-o lista, pentru a putea face referire rapida la ele
    def parameters(self):
        return [self.w_h, self.b_h, self.w_o, self.b_o]
    
    def forward(self,input_batch):
        # Intr-un MLP, intrarea este sub forma unui vector, deci un batch
        # este o matrice de dimensiunea nr_esantioane_batch x dimensiune esantion
        input_batch = torch.from_numpy(input_batch)
        self.hidden = torch.mm(input_batch, self.w_h) + self.b_h
        
        out = torch.mm(self.hidden, self.w_o) + self.b_o
        
        return out
    
# Instantiem reteaua
mlp = Retea_MLP(28*28,1000,10)



# Specificarea functiei loss
loss_function = nn.CrossEntropyLoss(reduction='sum')


train_data, train_labels = get_MNIST_train()
batch_size = 128 # Se poate si mai mult in cazul curent, dar este o valoare frecventa
nr_epoci = 15
nr_iteratii = train_data.shape[0] // batch_size # Din simplitate, vom ignora ultimul batch, daca este incomplet

optim = torch.optim.SGD(mlp.parameters(), lr=1e-5)

for ep in range(nr_epoci):
    predictii = []
    etichete = []

    for it in range(nr_iteratii):
        # Luam urmatoarele <batch_size> esantioane si etichete
        batch_data = train_data[it*batch_size : it*batch_size+batch_size, :]
        batch_labels = train_labels[it*batch_size : it*batch_size+batch_size]
        # Se calculeaza predictia retelei pentru datele curente (forward pass/ propagare inainte)
        current_predict = mlp.forward(batch_data)

        # Se calculeaza valoarea momentana a functiei loss
        loss = loss_function(current_predict, torch.from_numpy(batch_labels)) 
        
        # Se memoreaza predictiile si etichetele aferente batch-ului actual (pentru calculul acuratetii)
        current_predict = np.argmax(current_predict.detach().numpy(), axis=1)
        predictii = np.concatenate((predictii,current_predict))
        etichete = np.concatenate((etichete,batch_labels))
        
        # Antrenarea propriu-zisa
        
            # 1. Se sterg toti gradientii calculati anteriori, pentru toate variabilele antrenabile
            # deoarece, metoda <backward> acumuleaza noile valori, in loc sa le inlocuiasca.
        optim.zero_grad()
            # 2. Calculul tuturor gradientilor. Backpropagation
        loss.backward()
            # 3. Actualizarea tuturor ponderilor, pe baza gradientilor.
        optim.step()
        
        

    # Calculam acuratetea
    acc = np.sum(predictii==etichete)/len(predictii)
    print( 'Acuratetea la epoca {} este {}%'.format(ep+1,acc*100) )

    # Se genereaza o permutare noua a tuturor esantioanelor si etichetelor corespunzatoare
    perm = np.random.permutation(train_data.shape[0])
    train_data = train_data[perm,:]
    train_labels = train_labels[perm]

# ~ 8. Testarea retelei

# ~ Odata terminata antrenarea retelei, putem testa pe un set de date noi. Veti observa ca structura de la bucla de antrenare ne va ajuta in continuare:

test_data, test_labels = get_MNIST_test()
batch_size_test = 100 # pentru usurinta, ca sa testam toate etichetele alegem un divizor al numarului de date de test
nr_iteratii_test = test_data.shape[0] // batch_size_test
    
predictii = []
for it in range(nr_iteratii_test):
    batch_data = test_data[it*batch_size_test : it*batch_size_test+batch_size_test, :]
    batch_labels = test_labels[it*batch_size_test : it*batch_size_test+batch_size_test]

    current_predict = mlp.forward(batch_data)
    current_predict = np.argmax(current_predict.detach().numpy(),axis=1)
    predictii = np.concatenate((predictii,current_predict))

acc = np.sum(predictii==test_labels)/len(predictii)
print( 'Acuratetea la test este {}%'.format(acc*100) )

