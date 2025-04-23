import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter



class VaDE_new(torch.nn.Module):
    def __init__(self, in_dim=784, latent_dim=10, n_classes=10):
        super(VaDE_new, self).__init__()

        self.pi_prior = Parameter(torch.ones(n_classes)/n_classes)  # c,先验概率
        self.mu_prior = Parameter(torch.zeros(n_classes, latent_dim))  # 10个类别对应的均值
        self.log_var_prior = Parameter(torch.randn(n_classes, latent_dim))  # 10个类别对应的方差
        
        self.fc1 = nn.Linear(in_dim, 800) #Encoder
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 2048)

        self.mu = nn.Linear(2048, latent_dim) #Latent mu  # 制定了隐层的维度
        self.log_var = nn.Linear(2048, latent_dim) #Latent logvar

        self.fc4 = nn.Linear(latent_dim, 2048) 
        self.fc5 = nn.Linear(2048, 500)
        self.fc6 = nn.Linear(500, 800)
        self.fc7 = nn.Linear(800, in_dim) #Decoder

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.mu(h), self.log_var(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        return self.fc7(h) # 作用不是很大吧F.sigmoid(self.fc7(h))

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z


class Autoencoder(torch.nn.Module):
    def __init__(self, in_dim=784, latent_dim=10):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, 800) #Encoder
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 2048)
        # self.fc1 = nn.Linear(in_dim, 512) #Encoder
        # self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(512, 2048)

        self.mu = nn.Linear(2048, latent_dim) #Latent code
        self.log_var = nn.Linear(2048, latent_dim)  # Latent logvar

        # self.fc4 = nn.Linear(latent_dim, 2048)
        # self.fc5 = nn.Linear(2048, 512)
        # self.fc6 = nn.Linear(512, 512)
        # self.fc7 = nn.Linear(512, in_dim) #Decoder
        self.fc4 = nn.Linear(latent_dim, 2048)
        self.fc5 = nn.Linear(2048, 500)
        self.fc6 = nn.Linear(500, 800)
        self.fc7 = nn.Linear(800, in_dim) #Decoder

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.mu(h), self.log_var(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        return self.fc7(h)  # 作用不是很大吧F.sigmoid(self.fc7(h))

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z
    # def encode(self, x):
    #     h = F.relu(self.fc1(x))
    #     h = F.relu(self.fc2(h))
    #     h = F.relu(self.fc3(h))
    #     return self.mu(h)
    #
    # def decode(self, z):
    #     h = F.relu(self.fc4(z))
    #     h = F.relu(self.fc5(h))
    #     h = F.relu(self.fc6(h))
    #     return self.fc7(h)  #F.sigmoid(self.fc7(h))
    #
    # def forward(self, x):
    #     z = self.encode(x)
    #     x_hat = self.decode(z)
    #     return x_hat