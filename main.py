import torch 
from torch.utils.data import DataLoader, random_split
from BanknoteDateset import BanknoteDataset
from BanknoteLoss import BanknoteLoss
from BanknoteOptimizer import BanknoteOptimizer
from BanknoteModel import BanknoteModel
from sklearn.model_selection import  train_test_split
import numpy as np

def train_banknot():

    torch.manual_seed(42) ###for being  reproducible program 

    ### create object of dataset 
    banknotedataset = BanknoteDataset("banknote-authentication_csv.csv")

    ### prepare number of train data and test and validation 
    lengthss = [int(len(banknotedataset)* 0.7) ,int(len(banknotedataset)* 0.2),\
    int(len(banknotedataset)* 0.1)+1 ]

    train_data, test_data, validation_data = random_split(banknotedataset, lengthss)

    # ### prepare data 
    # X, X_test, y, y_test  = train_test_split(\
    #     banknotedataset.data, banknotedataset.labels, test_size=0.2 , random_state=42)

    # X_train, X_valid, y_train, y_valid = train_test_split( \
    #     X, y, test_size=0.1 , random_state=42)

   
    # y_train = y_train.reshape(-1,1)
    # print(np.shape(X_train))
    # print(np.shape(y_train))

    # train_data = np.concatenate((X_train, y_train), axis=1)


   

    dataloader_train = DataLoader(train_data.dataset, batch_size=10, shuffle=True)
    dataloader_validation = DataLoader(validation_data.dataset, batch_size=10, shuffle=True)
    dataloader_test = DataLoader(test_data.dataset, batch_size=10, shuffle=True)

    ### hyperparameter 
    lr=0.01
    num_epoch = 30
    input_dim = banknotedataset.input_dim
    output_dim = 1

    model = BanknoteModel(input_dim, output_dim)
    


    # print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
    # print(model.fc1.weight.requires_grad_)   
    # print(model.fc2.weight.requires_grad_)   
    
    ### set loss function and optimizer 

    # CentropyLoss = BanknoteLoss()


    loss = torch.nn.BCELoss()
  
    # optimizer = BanknoteOptimizer(model.parameters, lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    ### train model )
    # iter_dataloader_validation = iter(dataloader_validation)

    # number_of_batch_validation =len(dataloader_validation)
    # counter_valid=0




    for epoch in range(num_epoch):

        for (data_batch, label_batch) in dataloader_train:
            model.train()
            optimizer.zero_grad()
            out = model(data_batch)
            labels = label_batch.view(-1,1)
            
            # out = (out>=0.5).float()   #### lazem nist taghir koneh? va aya age tageer\????
            # bedam requires_grad ham false mish???
            # _, predicted = torch.max(out, 1)

            loss_value =loss(out, labels)
            loss_value.backward()

            optimizer.step()

        if epoch % 9 == 0:
            with torch.no_grad():
                model.eval()
                (data, label) = next(iter(dataloader_validation))
                out = model(data)
                label_val = label.view(-1,1)
                loss_value_valid=loss(out, label_val)
                print('Epoch [%d/%d] Train Loss: %.4f' % (epoch , num_epoch,\
                loss_value.item()))
                print('Epoch [%d/%d] Validation Loss: %.4f' % (epoch , num_epoch, \
                loss_value_valid.item()))

    
# with torch.no_grad        
   







 













    









def evaluate_banknote():
    pass

if __name__ == "__main__":
    train_banknot()

  
    
    # print(sys.path)

        