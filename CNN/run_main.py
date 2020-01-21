from main import train
from main import test
from Model import *
from utils import *
from dataset import CatDogDataset

image_size = (100, 100)
image_row_size = image_size[0] * image_size[1]

def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                 transforms.Resize(image_size),
                                 transforms.Grayscale(),
                                transforms.ToTensor(), 
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize(mean, std)])


path = '/home/aims/Documents/Tutors/Nando_assignment/data 1/train'
dataset = CatDogDataset(path, transform=transform)

path1 = '/home/aims/Documents/Tutors/Nando_assignment/data 1/val'
dataset1 = CatDogDataset(path1, transform=transform)

### Train
l1= dataset.__len__()

print('the len of the train data is \n {}'.format(l1))

### Test
l2= dataset1.__len__()

print('the len of the train data is \n {}'.format(l2))

### For the train data
shuffle     = True
batch_size  = 4
num_workers = 1
dataloader  = DataLoader(dataset=dataset, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)

### For the train data
shuffle     = True
batch_size  = 4
num_workers = 1
dataloader1  = DataLoader(dataset=dataset1, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)


input_size  = 100*100*3   # images are 28x28 pixels
output_size = 2
n_feature =3

#### Change of names
train_loader = dataloader
test_loader = dataloader1

model = CNN(input_size, n_feature, output_size)
model2 = CNN1(input_size, n_feature, output_size)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
optimizer1 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model)))

#### Change of names
train_loader = dataloader
test_loader = dataloader1

print('Here we are using the First CNN of the file Model.py')

for epoch in range(0, 1):
    train(epoch, model,train_loader,optimizer)
    test(model,test_loader)
    
### Model 2


print('Here we are using the second CNN2 of the file Model.py')


for epoch in range(0, 1):
    train(epoch, model2,train_loader,optimizer1)
    test(model2,test_loader)