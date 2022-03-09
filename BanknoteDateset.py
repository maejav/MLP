import torch 
from  torch.utils.data import DataLoader ,Dataset
import pandas as pd

class BanknoteDataset(Dataset):
    def __init__(self, Url):
        super(BanknoteDataset, self).__init__()
        self.dataframe = pd.read_csv(Url)
        # self.traindata =##
        # self.evaluationdata=##
        # self.testdata = ##

        print("number of features:",len(self.dataframe.columns))
        self.input_dim = len(self.dataframe.columns)-1 ####
        print("input dimension :", self.input_dim)
        raw_data =  self.dataframe.iloc[:, :self.input_dim ]
        self.data = raw_data.values
        # self.data[:,self``.input_dim] = self.data[:,self.input_dim]-1
        print("we are in original data")

        print(self.data)
        raw_labels = self.dataframe.iloc[:, self.input_dim]
        self.labels = raw_labels.values - 1
        print(self.labels)





    





        



      
