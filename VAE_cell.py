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