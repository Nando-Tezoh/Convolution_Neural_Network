from utils import *

accuracy_list = []

def train(epoch, model,train_loader,optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # permute pixels
        #print(target.shape)
        #print('data\n', data.shape)


        #data = data.view(-1, 100*100*1)
        #data = data[:, perm]
        data = data.view(-1, 3, 100, 100)
        
        optimizer.zero_grad()
        output = model(data)
        #print(output.size())
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # permute pixels
        #data = data.view(-1, 100*100*3)
        #data = data[:, perm]
        data = data.view(-1, 3, 100, 100)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    print('Accuracy \n'.format(accuracy))
    #print(pred)