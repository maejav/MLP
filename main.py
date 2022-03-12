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
    number_of_test = int(len(banknotedataset)* 0.2)
    number_of_train = int(len(banknotedataset)* 0.7)
    number_of_valid = int(len(banknotedataset)* 0.1) + 1

    lengthss = [int(len(banknotedataset)* 0.7) ,int(len(banknotedataset)* 0.2),\
    int(len(banknotedataset)* 0.1)+1 ]

    train_data, test_data, validation_data = random_split(banknotedataset, lengthss)

    # ### prepare data  first approach
    # X, X_test, y, y_test  = train_test_split(\
    #     banknotedataset.data, banknotedataset.labels, test_size=0.2 , random_state=42)

    # X_train, X_valid, y_train, y_valid = train_test_split( \
    #     X, y, test_size=0.1 , random_state=42)

   
    # y_train = y_train.reshape(-1,1)
    # print(np.shape(X_train))
    # print(np.shape(y_train))

    # train_data = np.concatenate((X_train, y_train), axis=1)



    dataloader_train = DataLoader(train_data, batch_size=10, shuffle=True)
    dataloader_validation = DataLoader(validation_data, batch_size=10, shuffle=True)
    dataloader_test = DataLoader(test_data, batch_size=10, shuffle=True)

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

  
   


    for epoch in range(num_epoch):

        model.train()
        numcorrect_train = 0
        numcorrect_valid = 0
        totalloss_train = 0

        for (data_batch, label_batch) in dataloader_train:
            
            optimizer.zero_grad()
            out_train = model(data_batch)
            labels_train = label_batch.view(-1,1)
            loss_value_train =loss(out_train, labels_train)
            loss_value_train.backward()
            optimizer.step()
            if epoch % 3 == 0 :
                # print("EEEEEEEEEEEEEEEEE", epoch)
                # print(numcorrect_train)
                totalloss_train = totalloss_train + loss_value_train.item()
                temp = (out_train>=0.5).float() 
                temp = torch.flatten(temp)
                # print("!!!!!!!!!!!! : ", torch.sum(torch.eq(label_batch, temp)).item().__int__())
                numcorrect_train= numcorrect_train + \
                torch.sum(torch.eq(label_batch, temp)).item().__int__()
        # print("number of correct train data:", numcorrect_train)
        # print(number_of_train)
        numcorrect_train = numcorrect_train / number_of_train
        # print()
        totalloss_train = totalloss_train / len(dataloader_train)     
        totalloss_val = 0
        if epoch % 3 == 0:
            with torch.no_grad():
                model.eval()
                for (data , label) in dataloader_validation:
                    out_validation = model(data)
                    label_val = label.view(-1,1)
                    loss_value_valid=loss(out_validation, label_val)
                    totalloss_val = totalloss_val + loss_value_valid.item()
                    temp = (out_validation>=0.5).float() 
                    temp = torch.flatten(temp)
                # print("!!!!!!!!!!!! : ", torch.sum(torch.eq(label_batch, temp)).item().__int__())
                    numcorrect_valid= numcorrect_valid + \
                    torch.sum(torch.eq(label, temp)).item().__int__()
                numcorrect_valid = numcorrect_valid / number_of_valid                   
                totalloss_val= totalloss_val / len(validation_data )
                print('Epoch [%d/%d] Train Loss: %.4f Train Accuracy: %.2f ' % (epoch , num_epoch,\
                totalloss_train, numcorrect_train*100 ))
                print('Epoch [%d/%d] Validation Loss: %.4f Validation Accuracy: %.2f' % (epoch , num_epoch, \
                totalloss_val, numcorrect_valid*100 ))


    ### test model  
    numcorrect_test = 0
    with torch.no_grad():
        model.eval()
        for (data_test, label_test) in dataloader_test :
            out_test = model(data_test)
            out_test = (out_test>=0.5).float() 
            out_test = torch.flatten(out_test)
            temp = torch.sum(torch.eq(label_test, out_test)).item().__int__()
            numcorrect_test = numcorrect_test + temp

       
        
        print('Accuracy of the network on Test Data %.2f %%'\
        % (100 * ( numcorrect_test/ number_of_test).__round__(4)))






def evaluate_banknote():
    pass

if __name__ == "__main__":
    train_banknot()

  
    
    # print(sys.path)

        