import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
from torchvision import datasets


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, latent_num=2, device=None):
        self._device = device
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        self.mean_layer = nn.Linear(latent_dim, latent_num)
        self.logvar_layer = nn.Linear(latent_dim, latent_num)

        self.decode_input = nn.Sequential(
            nn.Linear(latent_num, latent_dim),
            nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def sampling(self, mean, var):
        epsilon = torch.randn_like(var).to(self._device)
        z = mean + var*epsilon
        return z

    def decode(self, x):
        x = self.decode_input(x)
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sampling(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


def loss_function(x, x_hat, mean, logvar):
    mse = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return mse + kld


def train(model, data_loader, batch_size, optimizer, epochs, input_dim, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.view(batch_size, input_dim).to(device)
            x_hat, mean, logvar = model(x)
            loss = loss_function(x, x_hat, mean, logvar)
            overall_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch:', epoch+1, '\tAve loss:', overall_loss/(batch_idx*batch_size))


def main():
    data_path = 'datasets'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(data_path, train=True, transform=transform, download=True)

    batch_size = 100
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim=784
    latent_num = 2 # 2-d
    model = VAE(input_dim=input_dim, latent_num=latent_num, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 300
    train(model, train_loader, batch_size, optimizer, epochs, input_dim, device)

    # sampling 2-d
    sample1, sample2 = 0.1, 0.5
    z_sample = torch.tensor([[sample1, sample2]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(28, 28)


if __name__ == '__main__':
    main()