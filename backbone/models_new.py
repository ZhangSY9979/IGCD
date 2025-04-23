import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class VaDE_new(torch.nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[800, 500], latent_dim=10, n_classes=10):
        super(VaDE_new, self).__init__()

        self.pi_prior = Parameter(torch.ones(n_classes) / n_classes)  # c,先验概率
        self.mu_prior = Parameter(torch.zeros(n_classes, latent_dim))  # 10个类别对应的均值
        self.log_var_prior = Parameter(torch.randn(n_classes, latent_dim))  # 10个类别对应的方差

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # Encoder
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.fc3 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.mu = nn.Linear(hidden_dims[1], latent_dim)  # Latent mu  # 制定了隐层的维度
        self.log_var = nn.Linear(hidden_dims[1], latent_dim)  # Latent logvar

        self.fc4 = nn.Linear(latent_dim, hidden_dims[1])
        self.fc5 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc6 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.fc7 = nn.Linear(hidden_dims[0], input_dim)  # Decoder

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


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[800, 500], latent_dim=10, n_classes=10):
        super(Autoencoder, self).__init__()


        self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # Encoder
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.fc3 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.mu = nn.Linear(hidden_dims[1], latent_dim)  # Latent mu  # 制定了隐层的维度
        self.log_var = nn.Linear(hidden_dims[1], latent_dim)  # Latent logvar

        self.fc4 = nn.Linear(latent_dim, hidden_dims[1])
        self.fc5 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc6 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.fc7 = nn.Linear(hidden_dims[0], input_dim)  # Decoder

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