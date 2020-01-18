from utils import *

class CNN(nn.Module):
    def __init__(self, input_size, n_feature, output_size):
        super(CNN, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=(5,5))
        self.fc1 = nn.Linear(22*22*n_feature, 150)
        self.fc2 = nn.Linear(150, 2)
        
    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2,2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2,2))
        x = x.view(-1, 22*22*self.n_feature)#(64,22*22*3)
 
        x = self.fc1(x)
       
        x = F.relu(x)
        x = self.fc2(x)
     
        x = F.log_softmax(x, dim=1)
        return x
    
    
    
class CNN1(nn.Module):
    
    def __init__(self, input_size, n_feature, output_size):
        super(CNN1, self).__init__()
        
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(10, n_feature, kernel_size=(5,5))
        self.conv3 = nn.Conv2d(n_feature, n_feature, kernel_size=(5,5))
        self.fc1 = nn.Linear(9*9*n_feature, 150)
        self.fc2 = nn.Linear(150, 2)
        
        
    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2,2))
        

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2,2))
        
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size = (2,2))
        
        x = x.view(-1, 9*9*self.n_feature)#(64,22*22*3)
       
        x = self.fc1(x)
      
        x = F.relu(x)
        x = self.fc2(x)
        
        x = F.log_softmax(x, dim=1)
       
        return x
    
        
        
    