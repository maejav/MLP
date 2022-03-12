import torch 
from  torch.utils.data import DataLoader ,Dataset
import pandas as pd

class BanknoteDataset(Dataset):
    def __init__(self, Url):

        super(BanknoteDataset, self).__init__()
        self.dataframe = pd.read_csv(Url).set_index('V1')
        self.input_dim = len(self.dataframe.columns)-1 #### input dimension except labels 
        all_data = self.dataframe.iloc[:, :self.input_dim+1]
        all_data.iloc[:, self.input_dim] = all_data.iloc[:, self.input_dim]-1
        self.all_data = all_data.values
        raw_data =  self.dataframe.iloc[:, :self.input_dim ] ### input date exccept labels
        self.data = raw_data.values
        raw_labels = self.dataframe.iloc[:, self.input_dim]
        self.labels = raw_labels.values - 1 ### convert to binary form for classification 
        self.all_data = torch.FloatTensor(self.all_data)
        self.data = torch.FloatTensor(self.data)
        self.labels = torch.FloatTensor(self.labels)

        self.x = torch.FloatTensor(self.data)
        self.y = torch.FloatTensor(self.labels)


    def __getitem__(self, index):
        # if index <= len(self.data):
        #     item = (self.data[index, :], self.labels[index])
        #     return item
        # else :
        #     print("Index is out of range: ")
        #     return False

        ### or 

        try:
            item = (self.data[index, :], self.labels[index])
            return item
        except:
            print("Index is out of range")
            return False
    

    def __len__(self):
        return len(self.data) ### recommand to write len(self.labels) it is an important point 










    





        



      
