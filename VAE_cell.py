#Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from tqdm.notebook import tqdm
from torchvision.utils import save_image


#%%
# dimensions of latent space
zdim = 10


# Variational Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder
        self.fc1 = nn.Linear(128 * 128, 64*64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64 * 64, 350)
        self.fc2m = nn.Linear(350, zdim)  # mu layer
        self.fc2s = nn.Linear(350, zdim)  # sd layer

        # decoder
        self.fc3 = nn.Linear(zdim, 64 * 64)
        self.fc4 = nn.Linear(64 * 64, 350)
        self.fc5 = nn.Linear(350, 128 * 128)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc2m(h2), self.fc2s(h2)

    # reparameterize
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.relu(self.fc4(h3))
        return self.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 128 * 128))
        z = self.reparameterize(mu, logvar)
        val = self.decode(z)
        return val, mu, logvar
#%%

# loss function for VAE are unique and use Kullback-Leibler
# divergence measure to force distribution to match unit Gaussian
def loss_function(recon_x, x, mu, logvar):
    #     print(x.view(-1, 28 * 28))
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 128 * 128))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= 64 * 128 * 128
    return bce + kld


def train(model, num_epochs=1, batch_size=64, learning_rate=1e-3):
    model.train()  #train mode
    torch.manual_seed(42)

    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), learning_rate)

    for epoch in tqdm(range(num_epochs)):
        for data in train_loader:  # load batch
            img, _ = data
            img = img.to(device)

            recon, mu, logvar = model(img)
            loss = loss_function(recon, img, mu, logvar)  # calculate loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))


"""
A Convolutional Variational Autoencoder
"""
class ConvolutionalVAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=32*116*116, zDim=20):
        super(ConvolutionalVAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 7)
        self.encConv2 = nn.Conv2d(16, 32, 7)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)
        self.flatten = nn.Flatten()

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 7)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 7)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 32*116*116)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 116, 116)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar
#%%

if __name__ == "__main__":
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor()
    ])

    # train data
    TRAIN_DATA_PATH = "./filter_extracted_cells/train/"
    train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=64, shuffle=True)

    # test data
    TEST_DATA_PATH = "./filter_extracted_cells/test/"
    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if True:
        model = Autoencoder()
        model = model.to(device)
        model.load_state_dict(torch.load("./vae_cell.pt"))
    else:
        model = Autoencoder()
        model = model.to(device)
        train(model, num_epochs=100, batch_size=64, learning_rate=0.005)