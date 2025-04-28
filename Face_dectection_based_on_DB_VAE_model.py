
# %%
import comet_ml
COMET_API_KEY = "Fho0NbqmvO0HuZr2yC3PqWhYy"
import mitdeeplearning as mdl


# %%
import os
import random
import IPython
import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

device = torch.device("cuda")
cudnn.benchmark = True


# %%
# Get the dataset
CACHE_DIR = Path.home() / ".cache" / "mitdeeplearning"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

path_to_training_data = CACHE_DIR.joinpath("train_face.h5")

if path_to_training_data.is_file():
    print(f"Using cached training data from {path_to_training_data}")
else:
    print(f"Downloading training data to {path_to_training_data}")
    url = "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
    torch.hub.download_url_to_file(url, path_to_training_data)

channels_last = False
loader = mdl.lab2.TrainingDatasetLoader(
    path_to_training_data, channels_last=channels_last
)


# %%
number_of_training_examples = loader.get_train_size()
(images, labels) = loader.get_batch(100)


# %%
# Batch, channel, height, width
B, C, H, W = images.shape


# %%
# Check the contents of the dataset
face_images = images[np.where(labels == 1)[0]].transpose(0,2,3,1)
not_face_images = images[np.where(labels == 0)[0]].transpose(0,2,3,1)

idx_face = 23
idx_not_face = 9

plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(face_images[idx_face])
plt.title('Face')
plt.grid(False)

plt.subplot(1,2,2)
plt.imshow(not_face_images[idx_not_face])
plt.grid(False)


# %%
n_filters = 12
in_channels = images.shape[1]

def make_standard_classifier(n_outputs):
    """Create a standard CNN model"""
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0):
            super().__init__()
            self.conv_relu_bn = nn.Sequential(nn.Conv2d(in_channels,
                                              out_channels, kernel_size, stride, padding),
                                              nn.ReLU(inplace=True),
                                              nn.BatchNorm2d(out_channels))
        
        def forward(self, x):
            x = self.conv_relu_bn(x)
            return x
    
    model = nn.Sequential(ConvBlock(in_channels, n_filters, kernel_size=5, stride=2, padding=2),
                          ConvBlock(n_filters, 2*n_filters, kernel_size=5, stride=2, padding=2),
                          ConvBlock(2*n_filters, 4*n_filters, kernel_size=3, stride=2, padding=1),
                          ConvBlock(4*n_filters, 6*n_filters, kernel_size=3, stride=2, padding=1),
                          nn.Flatten(),
                          nn.Linear(H // 16 * W // 16 * 6 * n_filters, 512),
                          nn.ReLU(inplace=True),
                          nn.Linear(512, n_outputs))
    return model

standard_classifier = make_standard_classifier(n_outputs=1)


# %%
print(standard_classifier)


# %%
# Create a comet experiment
def create_experiment(project_name, params):
    if "experiment" in locals():
        experiment.end()
    experiment = comet_ml.Experiment(api_key=COMET_API_KEY, project_name=project_name)
    for param, value in params.items():
        experiment.log_parameter(param, value)
    experiment.flush()

    return experiment


# %%
print(loader)

# %%
# Train CNN model
loss_fn = nn.BCEWithLogitsLoss()
params = dict(batch_size=32, num_epochs = 2, learning_rate = 5e-4)

experiment = create_experiment("6S191_lab2_part2_CNN", params)
optimizer = optim.Adam(standard_classifier.parameters(), lr = params['learning_rate'])
loss_history = mdl.util.LossHistory(smoothing_factor=0.99)
plotter = mdl.util.PeriodicPlotter(sec=2, scale = 'semilogy')
if hasattr(tqdm, '_instances'): tqdm._instances.clear()
standard_classifier.to(device)
standard_classifier.train()

def standard_train_step(x, y):
    x = torch.from_numpy(x).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    optimizer.zero_grad()
    logits = standard_classifier(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()

    return loss

step = 0
for epoch in range(params["num_epochs"]):
    for idx in tqdm(range(loader.get_train_size() // params["batch_size"])):
        x, y = loader.get_batch(params["batch_size"])
        loss = standard_train_step(x, y)
        loss_value = loss.detach().cpu().numpy()

        loss_history.append(loss_value)
        plotter.plot(loss_history.get())

        experiment.log_metric("loss", loss_value, step=step)
        step += 1

experiment.end()


# %%
# Evaluate CNN model
standard_classifier.eval()

(batch_x, batch_y) = loader.get_batch(5000)
batch_x = torch.from_numpy(batch_x).float().to(device)
batch_y = torch.from_numpy(batch_y).float().to(device)

with torch.inference_mode():
    y_pred_logits = standard_classifier(batch_x)
    y_pred_standard = torch.round(torch.sigmoid(y_pred_logits))
    acc_standard = torch.mean((batch_y == y_pred_standard).float())

print("Standard CNN accuracy on (potentially biased) training set: {:.4f}".format(acc_standard.item()))


# %%
test_faces = mdl.lab2.get_test_faces(channels_last=channels_last)
keys = ["Light Female", "Light Male", "Dark Female", "Dark Male"]

fig, axs = plt.subplots(1, len(keys), figsize=(7.5, 7.5))
for i, (group, key) in enumerate(zip(test_faces, keys)):
    axs[i].imshow(np.hstack(group).transpose(1,2,0))
    axs[i].set_title(key, fontsize=15)
    axs[i].axis("off")


# %%
standard_classifier_probs_list = []

with torch.inference_mode():
    for x in test_faces:
        x = torch.from_numpy(np.array(x, dtype=np.float32)).to(device)
        logits = standard_classifier(x)
        probs = torch.squeeze(torch.sigmoid(logits), dim=-1)
        standard_classifier_probs_list.append(probs.cpu().numpy())
standard_classifier_probs = np.stack(standard_classifier_probs_list, axis=0)

xx = range(len(keys))
yy = standard_classifier_probs.mean(axis=1)
plt.bar(xx, yy)
plt.xticks(xx, keys)
plt.ylim(max(0, yy.min() - np.ptp(yy) / 2.0), yy.max() + np.ptp(yy) / 2.0)
plt.title("Standard classifier predictions")


# %%
def vae_loss_function(x, x_recon, mu, logsigma, kl_weight = 0.0005):
    """ Function to compute VAE loss.
    # Arguments:
        an input x,
        reconstructed output x_recon,
        encoded means mu,
        encoded log of variance logsigma,
        weight parameter for the latent loss kl_weight
    # Output:
        vae loss    
    """
    latent_loss = -0.5 * torch.sum(1 + logsigma - mu**2 - torch.exp(logsigma))
    reconstruction_loss = torch.mean(torch.abs(x-x_recon), dim = (1, 2, 3))
    vae_loss = kl_weight * latent_loss + reconstruction_loss
    return vae_loss

def sampling(z_mean, z_logsigma):
    """ Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments:
        z_mean, z_logsigma (tensor): mean and log of variance of latent distribution (Q_z|X)
    # Output:
        z (tensor): sampled latent vector
    """
    eps = torch.randn_like(z_mean)
    z = z_mean + eps*torch.exp(0.5*z_logsigma)
    return z

def debiasing_loss_function(x, x_pred, y, y_logit, mu, logsigma):
    """ Loss function for DB-VAE.
    # Arguments:
        x: true input
        x_pred: reconstructed output
        y: true labels
        y_logit: predicted labels
        mu: mean of latent distribution (Q_z|X)
        logsigma: log of variance of latent distribution (Q_z|X)
    # Return:
        total_loss: DB_VAE total loss
        classification_loss = DB_VAE classification loss
    """
    vae_loss = vae_loss_function(x, x_pred, mu, logsigma)

    classification_loss = F.binary_cross_entropy_with_logits(y_logit, y, reduction='none')

    y = y.float()
    face_indicator = (y == 1.0).float()
    total_loss = torch.mean(classification_loss * face_indicator + vae_loss)
    return total_loss, classification_loss


# %%
n_filters = 12
latent_dim = 100

def make_face_decoder_network(latent_dim = 100, n_filters = 12):
    """ Function builds a face decoder network.
    # Arguments:
        latent_dim (int): the dimension of the latent distribution (Q_z|X)
        n_filters (int): the number of convolutional filters
    # Returns:
        decoder_model (nn.Module): the decoder network
    """
    class FaceDecoder(nn.Module):
        def __init__(self, latent_dim, n_filters):
            super().__init__()
            self.latent_dim = latent_dim
            self.n_filters = n_filters
            self.linear = nn.Sequential(
                nn.Linear(self.latent_dim, 4 * 4 * 6 * self.n_filters), nn.ReLU()
            )

            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=6 * self.n_filters,
                    out_channels=4 * self.n_filters,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_channels=4 * self.n_filters,
                    out_channels=2 * self.n_filters,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                                   ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_channels=2 * self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_channels=self.n_filters,
                    out_channels=3,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1
                )
            )
        def forward(self, z):
            x = self.linear(z)
            x = x.view(-1, 6*self.n_filters, 4, 4)
            x = self.deconv(x)
            return x
    return FaceDecoder(latent_dim, n_filters)

# %%
# Define DB-VAE structure
class DB_VAE(nn.Module):
    def __init__(self, latent_dim = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = make_standard_classifier(n_outputs=2*self.latent_dim + 1)
        self.decoder = make_face_decoder_network()
    
    def encode(self, x):
        encoder_ouput = self.encoder(x)
        y_logit = encoder_ouput[:,0].unsqueeze(-1)
        z_mean = encoder_ouput[:, 1: self.latent_dim + 1]
        z_logsigma = encoder_ouput[:, self.latent_dim + 1 :]
        return y_logit, z_mean, z_logsigma
    
    def reparameterize(self, z_mean, z_logsigma):
        z = sampling(z_mean, z_logsigma)
        return z
    
    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction
    
    def forward(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)
        z = self.reparameterize(z_mean, z_logsigma)
        recon = self.decode(z)
        return y_logit, z_mean, z_logsigma, recon
    
    def predict(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)
        return y_logit
    
dbvae = DB_VAE(latent_dim)


# %%
# Function to return the means for an input image batch
def get_latent_mu(images, dbvae, batch_size=64):
    dbvae.eval()
    all_z_mean = []

    images_t = torch.from_numpy(images).float()

    with torch.inference_mode():
        for start in range(0, len(images_t), batch_size):
            end = start + batch_size
            batch = images_t[start:end]
            batch = batch.to(device).permute(0,3,2,1)
            _,z_mean,_,_ = dbvae(batch)
            all_z_mean.append(z_mean.cpu())
    
    z_mean_full = torch.concat(all_z_mean, dim=0)
    mu = z_mean_full.numpy()
    return mu


# %%
# Recompute the sampling probabilities for images within a batch based on how they distribute across the training data
def get_training_sample_probabilities(images, dbvae, bins=10, smoothing_fac=0.0001):
    print("Recomputing the sampling probabilites")

    mu = get_latent_mu(images, dbvae)

    training_sample_p = np.zeros(mu.shape[0], dtype=np.float64)

    for i in range(latent_dim):
        latent_distribution = mu[:, i]
        hist_density, bin_edges = np.histogram(
            latent_distribution, density=True, bins=bins
        )
        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')

        bin_idx = np.digitize(latent_distribution, bin_edges)

        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

        p = 1.0 / (hist_smoothed_density[bin_idx - 1])
        p = p / np.sum(p)

        training_sample_p = np.maximum(training_sample_p, p)
    
    training_sample_p /= np.sum(training_sample_p)

    return training_sample_p


# %%
# Train the DB-VAE model
params = dict(
    batch_size=32,
    learning_rate=5e-4,
    latent_dim=100,
    num_epochs = 2
)

experiment = create_experiment("6S191_lab2_part2_DBVAE", params=params)

dbvae = DB_VAE(params['latent_dim']).to(device)
optimizer = optim.Adam(dbvae.parameters(), lr=params['learning_rate'])

def debiasing_train_step(x, y):
    optimizer.zero_grad()

    y_logit, z_mean, z_logsigma, x_recon = dbvae(x)
    loss, class_loss = debiasing_loss_function(
        x, x_recon, y, y_logit, z_mean, z_logsigma
    )
    loss.backward()
    optimizer.step()
    return loss

all_faces = loader.get_all_train_faces()

step = 0
for i in range(params["num_epochs"]):
    IPython.display.clear_output(wait=True)
    print("Starting epoch {}/{}".format(i+1, params["num_epochs"]))
    p_faces = get_training_sample_probabilities(all_faces, dbvae)

    for j in tqdm(range(loader.get_train_size() // params['batch_size'])):
        (x, y) = loader.get_batch(params['batch_size'], p_pos=p_faces)
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)

        loss = debiasing_train_step(x, y)
        loss_value = loss.detach().cpu().numpy()
        experiment.log_metric('loss', loss_value, step=step)

        if j % 500 == 0:
            mdl.util.plot_sample(x, y, dbvae, backend='pt')
        
        step += 1

experiment.end()


# %%
# Evaluate DB-VAE model
dbvae_logits_list = []
for face in test_faces:
    face = np.asarray(face, dtype=np.float32)
    face = torch.from_numpy(face).to(device)

    with torch.inference_mode():
        logit = dbvae.predict(face)

    dbvae_logits_list.append(logit.detach().cpu().numpy())

dbvae_logits_array = np.concatenate(dbvae_logits_list, axis=0)
dbvae_logits_tensor = torch.from_numpy(dbvae_logits_array)
dbvae_probs_tensor = torch.sigmoid(dbvae_logits_tensor)
dbvae_probs_array = dbvae_probs_tensor.squeeze(dim=-1).numpy()

xx = np.arange(len(keys))

std_probs_mean = standard_classifier_probs.mean(axis=1)
dbvae_probs_mean = dbvae_probs_array.reshape(len(keys), -1).mean(axis=1)

plt.bar(xx, std_probs_mean, width=0.2, label='Standard CNN')
plt.bar(xx + 0.2, dbvae_probs_mean, width=0.2, label='DB-VAE')
plt.xticks(xx, keys)
plt.title("Network predictons on test dataset")
plt.ylabel("Probability")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.show()