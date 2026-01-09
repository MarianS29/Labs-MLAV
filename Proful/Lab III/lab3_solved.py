from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn

###### PLAN ######
# pasul 1: definim Dataset si reteaua
# pasul 1: initializarea elementelor importante: cai, variabile, Dataset, Dataloader, Retea, optimizator, loss
# pasul 2: training loop: un loop pentru epoci, un loop pentru iteratii

data_path = 'train-images.idx3-ubyte'
label_path = 'train-labels.idx1-ubyte'

batch_size = 128
epochs = 15

def get_MNIST_train(images_path, labels_path):
    
    mnist_train_data = np.zeros([60000,784])
    mnist_train_labels = np.zeros(60000)
    
    f = open(images_path,'r', encoding = 'latin-1')
    g = open(labels_path,'r', encoding = 'latin-1')
    
    byte = f.read(16) #4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
    byte_label = g.read(8) # 4bytes magic number, 4 bytes nr labels
    
    mnist_train_data = np.fromfile(f,dtype=np.uint8).reshape(60000,784)
    mnist_train_data = mnist_train_data.reshape(60000,1,28,28)
    mnist_train_labels = np.fromfile(g,dtype=np.uint8)
    
    # Conversii pentru a se potrivi cu procesul de antrenare    
    mnist_train_data = mnist_train_data.astype(np.float32)
    mnist_train_labels = mnist_train_labels.astype(np.int64)
    
    return mnist_train_data, mnist_train_labels

class DatasetMNIST(Dataset):
    def __init__(self, data_path, label_path):
        # in constructorul Dataset ar trebui sa existe toate datele, sau o cale de a accesa toate datele
        data, labels = get_MNIST_train(data_path, label_path)

        self.mnist_train_data = data
        self.mnist_train_labels = labels

    def __len__(self):
        return self.mnist_train_data.shape[0]
        
    def __getitem__(self, idx):
        # get item returneaza un singur sample (nu un batch)
        data_sample = self.mnist_train_data[idx, ...]
        label_sample = self.mnist_train_labels[idx, ...]

        one_sample = {'data':data_sample, 'label':label_sample}

        return one_sample

class simpleCNN(nn.Module):
    def __init__(self):
        #  in constructorul retelei (nn.Module) se definesc straturile si operatiile
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()


    def forward(self, x):
        # in forward sunt conectate si folosite
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1) # output-ul 2D al ultimei convolutii trebuie vectorizat pentru a se potrivi cu inputul stratului dens

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

training_dataset = DatasetMNIST(data_path, label_path)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # este doar un string care ia valoare 'cpu' sau 'cuda' in functie de capabilitatile hardware

model = simpleCNN().to(device)

loss_function = nn.CrossEntropyLoss(reduction='sum')

optim = torch.optim.SGD(model.parameters(), lr=1e-5)

       
for ep in range(epochs):
    
    all_predictions = []
    all_labels = []

    for batch in training_dataloader:

        batch_data = batch['data'].to(device)
        batch_labels = batch['label'].to(device)

        current_prediction = model.forward(batch_data)

        loss = loss_function(current_prediction, batch_labels)

        optim.zero_grad() # gradientii vechi sunt stersi
     
        loss.backward() # se calculeaza cei noi
        
        optim.step() # ponderile sunt actualizate  
        
        # la finalul iteratiei, se adauga predictiile si labelurile curente intr-un array mare, pentru a calcula o acuratete per epoca
        # alternativ, in cazul in care datele sunt mari, se logheaza acuratetea la fiecare iteratie si se mediaza
        current_prediction = current_prediction.detach().cpu().numpy() # conversie torch (cuda) tensor -> numpy array
        current_prediction = np.argmax(current_prediction, axis=1)
        batch_labels = batch_labels.cpu().detach().numpy()
        
        all_predictions = np.concatenate((all_predictions, current_prediction))
        all_labels = np.concatenate((all_labels, batch_labels))
        

    # Calculam acuratetea
    acc = np.sum(all_predictions==all_labels)/len(all_predictions)
    print( 'Acuratetea la epoca {} este {}%'.format(ep+1,acc*100) )
