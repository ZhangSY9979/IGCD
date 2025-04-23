import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class Classifier(nn.Module):
    def __init__(self, latent_dim, intermediate_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, num_classes)

    def forward(self, z):
        y = F.relu(self.fc1(z))
        y = F.softmax(self.fc2(y), dim=-1)
        return y

class VAE_cluster(torch.nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[800, 500], latent_dim=100, n_classes=10):
        super(VAE_cluster, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # Encoder
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.fc3 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.mu = nn.Linear(hidden_dims[1], latent_dim)  # Latent mu  # 制定了隐层的维度
        self.log_var = nn.Linear(hidden_dims[1], latent_dim)  # Latent logvar

        self.fc4 = nn.Linear(latent_dim, hidden_dims[1])
        self.fc5 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc6 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.fc7 = nn.Linear(hidden_dims[0], input_dim)  # Decoder

        self.classifer = Classifier(latent_dim, int(latent_dim/2), num_classes=n_classes)
        self.gaussian_mean = nn.Parameter(torch.zeros(n_classes, latent_dim))

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.mu(h), self.log_var(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        return self.fc7(h) # 作用不是很大吧  F.sigmoid

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def recon(self, x):
    #     mu, log_var = self.encode(x)
    #     x_hat = self.decode(mu)
    #     return x_hat

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        y = self.classifer(z)
        z_prior_mean = z.unsqueeze(1) - self.gaussian_mean.unsqueeze(0)
        # print(z_prior_mean.shape)
        return x_hat, mu, log_var, z, y, z_prior_mean

