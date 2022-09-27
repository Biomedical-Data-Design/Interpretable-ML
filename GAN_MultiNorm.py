import torch
from torch import nn
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
    def forward(self, x):
        output = self.model(x)
        return output

train_data_length = 6000
train_data = torch.zeros((train_data_length, 3))
rho01=0;rho02=0;rho12=0
sigma = [[1, rho01, rho02], [rho01, 1, rho12], [rho02, rho12, 1]]
mu = [2, 3, 4]
dstr = stats.multivariate_normal(mean=mu, cov=sigma)
x1=[];x2=[];x3=[]
for i in range(train_data_length):
    sample=dstr.rvs()
    x1.append(sample[0]);x2.append(sample[1]);x3.append(sample[2])
train_data[:, 0] = torch.tensor(x1)
train_data[:, 1] = torch.tensor(x2)
train_data[:, 2] = torch.tensor(x3)
train_labels = torch.zeros(train_data_length)
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

batch_size =75
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

discriminator = Discriminator()
generator = Generator()

lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, 3))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 3))

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

latent_space_samples = torch.randn(500000, 3)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()

# ep=0.1
# g_s_x1=[]
# for i in range(len(generated_samples)):
#     if (abs(generated_samples[i,1]-3)<=ep)&(abs(generated_samples[i,2]-4)<=ep):
#         g_s_x1.append(generated_samples[i,0])

l1=train_data[:, 1].tolist()
plt.subplot(121)
plt.hist(l1,100,density=True)

l2=generated_samples[:, 1].tolist()
plt.subplot(122)
plt.hist(l2,100,density=True)

# plt.subplot(223)
# plt.plot(g_s_x1)
plt.show()
# # plt.subplot(224)
# # plt.hist(g_s_x1,density=True)
# # plt.show()
# print(len(g_s_x1))
# print(g_s_x1)