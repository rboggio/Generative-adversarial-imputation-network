import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler



class Discriminator(nn.Module):

    def __init__(self, n_in, n_h1, n_h2, n_out):

        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.fc3 = nn.Linear(n_h2, n_out)
        self.weights_init()
    
    
    def weights_init(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [nn.init.xavier_normal_(layer.weight) for layer in layers]
        
        
    def forward(self, x, h):

        model = nn.Sequential(self.fc1,
                              nn.ReLU(),
                              self.fc2,
                              nn.ReLU(),
                              self.fc3,
                              nn.Sigmoid())  
        
        return model(torch.cat((x, h), dim=1))


class Generator(nn.Module):

    def __init__(self, n_in, n_h1, n_h2, n_out):

        super(Generator, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h1, bias=True)
        self.fc2 = nn.Linear(n_h1, n_h2, bias=True)
        self.fc3 = nn.Linear(n_h2, n_out, bias=True)
        self.weights_init()
    
    def weights_init(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [nn.init.xavier_normal_(layer.weight) for layer in layers]
        
        
    def forward(self, x, m):
        
        model = nn.Sequential(self.fc1,
                            nn.ReLU(),
                            self.fc2,
                            nn.ReLU(),
                            self.fc3,
                            nn.Sigmoid())  
        
        return model(torch.cat((x, m), dim=1))
       





def gain(data_x: np.array, data_test: np.array, hint_rate: float, batch_size: int, alpha: int, epochs:int):

    
    # Parameters
    batch_size = batch_size
    hint_rate = hint_rate
    alpha = alpha
    epochs = epochs
    no, dim = data_x.shape
    
    # Define mask matrix
    data_m = 1-np.isnan(data_x)

    # Normalization
    norm_data = MinMaxScaler().fit_transform(data_x)
    #norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)


    ## GAIN structure
    generator = Generator(n_in=dim*2, n_h1=256, n_h2=128, n_out=dim)
    discriminator = Discriminator(n_in=dim*2, n_h1=256, n_h2=128, n_out=dim)

    optimG = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimD = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    bce_loss = nn.BCELoss(reduction="mean")
    mse_loss = nn.MSELoss(reduction="mean")

    train_dataloader = DataLoader(norm_data_x, batch_size=batch_size, shuffle=False)
    train_dataloader_m = DataLoader(data_m, batch_size=batch_size, shuffle=False)




    for epoch in tqdm(range(epochs)):
        
        for batch_x, batch_m in zip(train_dataloader, train_dataloader_m):

          # Sample batch
          X_mb = batch_x
          M_mb = batch_m
          # Sample random vectors
          Z_mb = np.random.uniform(0, 0.01, size = [X_mb.shape[0], dim]) 
          # Sample hint vectors
          random_matrix = np.random.uniform(0., 1., size = [X_mb.shape[0], dim])
          binary_random_matrix = 1*(random_matrix < hint_rate)
          H_mb_temp = binary_random_matrix
          H_mb = M_mb * H_mb_temp

          # Combine random vectors with observed vectors
          X_tilda_mb = torch.tensor(M_mb * X_mb + (1-M_mb) * Z_mb).float()
          M_mb = torch.tensor(M_mb).float() 

          # Train Discriminator
          G_sample = generator(X_tilda_mb, M_mb)
          X_hat = (X_tilda_mb * M_mb) + (G_sample * (1-M_mb))
          D_prob = discriminator(X_hat, H_mb)
          D_loss = bce_loss(D_prob, M_mb)

          # re-init the gradients
          optimD.zero_grad()
          # perform back-propagation
          D_loss.backward()
          # update the weights
          optimD.step()
          


          # Train Generator
          G_sample = generator(X_tilda_mb, M_mb)
          X_hat = (X_tilda_mb * M_mb) + (G_sample * (1-M_mb))
          D_prob = discriminator(X_hat, H_mb)
          D_prob.detach_()
          G_loss1 = -torch.mean((1-M_mb) * torch.log(D_prob + 1e-8))
          G_mse_loss = mse_loss(M_mb*X_tilda_mb, M_mb*G_sample) / torch.mean(M_mb)
          G_loss = G_loss1 + alpha*G_mse_loss
          
          optimG.zero_grad()
          G_loss.backward()
          optimG.step()

          if epoch % 100 == 0:
            print(f'epoch {epoch}, loss {G_mse_loss.item()}') 
    

    no, dim = data_test.shape
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(data_test)
    X_test = np.nan_to_num(X_test, 0)

    M = 1-np.isnan(X_test)
    Z = np.random.uniform(0, 0.01, size = [no, dim])
    X_test_tilda = torch.tensor(M * X_test + (1-M) * Z).float()
    M = torch.tensor(M)
    G_sample = generator(X_test_tilda,M)
    # MSE Performance
    MSE_test_loss = torch.sum(((1-M) * X_test - (1-M)*G_sample)**2) / torch.sum(1-M)
    imputed_data = M * X_test + (1-M) * G_sample.detach().numpy()
    # Renormalization
    imputed_data = scaler.inverse_transform(imputed_data)
          
    
    return MSE_test_loss, imputed_data