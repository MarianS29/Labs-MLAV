from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

train_data_path = 'train-images.idx3-ubyte'
train_label_path = 'train-labels.idx1-ubyte'

validation_data_path = 't10k-images.idx3-ubyte'
validation_label_path = 't10k-labels.idx1-ubyte'

batch_size = 128
epochs = 15

def load_mnist(images_path, labels_path, num_samples):
    # alternativa mai corecta
    with open(images_path, 'rb') as f_img, open(labels_path, 'rb') as f_lbl:
        f_img.read(16)
        f_lbl.read(8)
        images = np.fromfile(f_img, dtype=np.uint8).reshape(num_samples, 1, 28, 28).astype(np.float32)
        labels = np.fromfile(f_lbl, dtype=np.uint8).astype(np.int64)
    return images, labels

class DatasetMNIST(Dataset):
    def __init__(self, data_path, label_path, num_samples):
        # in constructorul Dataset ar trebui sa existe toate datele, sau o cale de a accesa toate datele
        data, labels = load_mnist(data_path, label_path, num_samples=num_samples)

        self.mnist_train_data = data / 255.0
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

training_dataset = DatasetMNIST(train_data_path, train_label_path, num_samples=60000)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

validation_dataset = DatasetMNIST(validation_data_path, validation_label_path, num_samples=10000)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # este doar un string care ia valoare 'cpu' sau 'cuda' in functie de capabilitatile hardware

model = simpleCNN().to(device)

loss_function = nn.CrossEntropyLoss()

optim = torch.optim.SGD(model.parameters(), lr=1e-2)

train_acc_per_epoch = []
val_acc_per_epoch = []
       
for ep in range(epochs):
    
    per_batch_accuracy_list = []
    per_batch_validation_accuracy_list = []

    model.train()

    # training
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

        per_batch_accuracy = np.sum(current_prediction==batch_labels)/len(current_prediction)
        per_batch_accuracy_list.append(per_batch_accuracy)

    model.eval()

    # validation
    with torch.no_grad():
        for batch in validation_dataloader:

            batch_data = batch['data'].to(device)
            batch_labels = batch['label'].to(device)

            current_prediction = model.forward(batch_data)
        
            # la finalul iteratiei, se adauga predictiile si labelurile curente intr-un array mare, pentru a calcula o acuratete per epoca
            # alternativ, in cazul in care datele sunt mari, se logheaza acuratetea la fiecare iteratie si se mediaza
            current_prediction = current_prediction.detach().cpu().numpy() # conversie torch (cuda) tensor -> numpy array
            current_prediction = np.argmax(current_prediction, axis=1)
            batch_labels = batch_labels.cpu().detach().numpy()

            per_batch_validation_accuracy = np.sum(current_prediction==batch_labels)/len(current_prediction)
            per_batch_validation_accuracy_list.append(per_batch_validation_accuracy)


    per_epoch_accuracy = np.mean(per_batch_accuracy_list)
    per_epoch_validation_accuracy = np.mean(per_batch_validation_accuracy_list)

    train_acc_per_epoch.append(per_epoch_accuracy * 100)
    val_acc_per_epoch.append(per_epoch_validation_accuracy * 100)

    print(f'Acuratetea la train la epoca {ep+1} este {per_epoch_accuracy*100:.3f}')
    print(f'Acuratetea la validare la epoca {ep+1} este {per_epoch_validation_accuracy*100:.3f}')

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(7, 5))
plt.plot(epochs_range, train_acc_per_epoch, label='Train Accuracy')
plt.plot(epochs_range, val_acc_per_epoch, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()