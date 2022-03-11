# import torch 
# from torch.utils.data import DataLoader, random_split
# from BanknoteDateset import BanknoteDataset
# from BanknoteLoss import BanknoteLoss
# from BanknoteOptimizer import BanknoteOptimizer
# from BanknoteModel import BanknoteModel
# from sklearn.model_selection import  train_test_split
# import numpy as np


# if __name__ == "__main__" :




#     banknotedataset = BanknoteDataset("banknote-authentication_csv.csv")

#     ### test get item 

#     for i in range(banknotedataset.__len__()):
#         break
#         print(banknotedataset.__getitem__(i))


#     ### 
#     # 
#     ### test get len of data
#     print("length of all data:", banknotedataset.__len__())      


#     ### identify length of train test validation 

#     lengthss = [int(len(banknotedataset.all_data)* 0.7) \
#     ,int(len(banknotedataset.all_data)* 0.2),\
#     int(len(banknotedataset.all_data)* 0.1)+1 ]

#     train_data, test_data, validation_data = random_split(banknotedataset.all_data, lengthss)

#     dataloader = DataLoader(train_data.dataset,batch_sampler=True,collate_fn=True)

#     print("type of dataloader :", type(dataloader))
#     # print(dataloader)
   
#     dataset_iter = iter(train_data.dataset)

#     print("dataloader.dataset :", dataloader.dataset)


#     # print(dataloader.collate_fn)
#     # print(dataloader.batch_sampler)
#     # print(dataloader.collate_fn)

#     # for indices in dataloader.batch_sampler:
#     #     yield dataloader.collate_fn([next(dataset_iter) for _ in indices])
        

