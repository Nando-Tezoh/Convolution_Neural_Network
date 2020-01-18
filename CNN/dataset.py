####DATASET CODE

from utils import *

class CatDogDataset(Dataset):
    def __init__(self, path, transform=None):
        ## To create Classes
        self.classes= os.listdir(path)
        ### Create path for each Class
        self.path= [f"{path}/{className}" for className in self.classes]
        ### To copy the images (upload)
        self.file_list = [glob.glob(f"{x}/*") for x in self.path]
        
        ### applied some transformations on the image

        self.transform = transform
        ### append the corresponding
        files = []
        for i, className in enumerate(self.classes):
            for fileName in self.file_list[i]:
                files.append([i, className, fileName])
                
        ### Assign the files to file_list
        self.file_list = files
        ###delete files
        files = None
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fileName = self.file_list[idx][2]
        classCategory = self.file_list[idx][0]
        im = Image.open(fileName)
        if self.transform:
            im = self.transform(im)
        return im, classCategory#im.view(-1), classCategory