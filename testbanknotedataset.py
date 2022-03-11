from BanknoteDateset import BanknoteDataset

if __name__ == "__main__" :
    banknotedataset = BanknoteDataset("banknote-authentication_csv.csv")

    ### test get item 

    for i in range(banknotedataset.__len__()):
        print(banknotedataset.__getitem__(i))


    ### 
    # 
    ### test get len of data
    print("length of all data:", banknotedataset.__len__())       



