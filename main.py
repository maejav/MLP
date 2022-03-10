import torch 
from torch.utils.data import DataLoader
from BanknoteDateset import BanknoteDataset
from BanknoteLoss import BanknoteLoss
from BanknoteOptimizer import BanknoteOptimizer
from BanknoteModel import BanknoteModel
from sklearn.model_selection import  train_test_split
import numpy as np

def train_banknot():
    index = 4

    torch.manual_seed(42) ###for being  reproducible program 


    ### create object of dataset 
    banknotedataset = BanknoteDataset("banknote-authentication_csv.csv")



    ### get sample 
    (features, label) = banknotedataset.__getitem__(index)
    print((features, label))

    ### prepare data 
    X, X_test, y, y_test  = train_test_split(\
        banknotedataset.data, banknotedataset.labels, test_size=0.2 , random_state=42)

    X_train, X_valid, y_train, y_valid = train_test_split( \
        X, y, test_size=0.1 , random_state=42)

   
    y_train = y_train.reshape(-1,1)
    print(np.shape(X_train))
    print(np.shape(y_train))

    train_data = np.concatenate((X_train, y_train), axis=1)


    train_data = torch.tensor(train_data).float()  ### convert to torch 
    # train_data.requires_grad_= True
    # x_test = torch.tensor(X_test).float()
    # x_valid = torch.tensor(X_valid).float()
    # y_train = torch.tensor(y_train).float()
    # y_test = torch.tensor(y_test).float()
    # y_valid = torch.tensor(y_valid).float()

    dataloader = DataLoader(train_data, batch_size=10, shuffle=True)

    ### hyperparameter 
    lr=0.01
    num_epoch = 30
    input_dim = banknotedataset.input_dim
    output_dim = 1

    model = BanknoteModel(input_dim, output_dim)
    
    for param in model.parameters():
        param.requires_grad_= True


    print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
    print(model.fc1.weight.requires_grad_)   
    print(model.fc2.weight.requires_grad_)   
         

    ### set loss function and optimizer 

    # CentropyLoss = BanknoteLoss()
    loss = torch.nn.BCELoss()
    loss.requires_grad_ = True
    print(loss.requires_grad_)

    # optimizer = BanknoteOptimizer(model.parameters, lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    ### train model 

    for epoch in range(num_epoch):

        for  _, data_batch in enumerate(dataloader):
            optimizer.zero_grad()
            # print(data_batch[:, :4])


        # out=my_model(iris_dataset.x[iris_train.indices])
        # loss = criterion(out, iris_dataset.y[iris_train.indices])

            out = model(data_batch[:, :4])
            # print(data_batch)
            # print(out)
            # out = out.view(-1,1)
            # print(data_batch[:, 4])
            labels = data_batch[:, 4].view(-1,1)
            # print(labels)
            
            # prob = torch.tensor([0.3,0.4,0.6,0.7])
            # out = (prob>0.5).float()

            # out = (out>=0.5).float()   #### lazem nist taghir koneh? va aya age tageer\????
            # bedam requires_grad ham false mish???
            # print(out)

            
            

            # _, predicted = torch.max(out, 1)

            
            # predicted = predicted.view(-1)
            # target = data_batch[:,4].view(-1)
            # print(target.dim())
            # print(predicted.dim())  
            # out.requires_grad_=True
            # loss_value = loss(predicted, target)
            # # loss = CentropyLoss(predicted, data_batch[:,4])




            loss_value =loss(out, labels)
            

            # print(loss_value)
            # # CentropyLoss.backward()
            loss_value.backward()

            optimizer.step()
        if epoch % 5 == 0:
                print('Epoch [%d/%d] Train Loss: %.4f' % (epoch , num_epoch, loss_value.item()*100))
    

   







 













    









def evaluate_banknote():
    pass

if __name__=="__main__":
    train_banknot()

  
    
    # print(sys.path)

        