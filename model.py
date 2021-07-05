import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Gradient Reversal Layer, https://github.com/hanzhaoml/MDAN.git
class GradientReversalLayer(torch.autograd.Function):
    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input


# Multi-Layer Perceptrons
class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]

        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y


# MiniMaxCCA model for independence regularization
class MMDCCA(nn.Module):
    def __init__(self, view1_dim, view2_dim, phi_size, tau_size, latent_dim=1):
        super(MMDCCA, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.tau = MLP(view2_dim, tau_size, latent_dim)
        # gradient reversal layer
        self.grl1 = GradientReversalLayer()
        self.grl2 = GradientReversalLayer()

    def forward(self, x1, x2):
        y1 = self.phi(self.grl1(x1))
        y2 = self.tau(self.grl2(x2))

        return y1, y2


# Flatten 3D image
class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

# Unflatten to 3D image
class Unflatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], 64, 4, 4)
        return x

# Increase 2 dims to get a 3D tensor
class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


# CNNEncoder (the input is fixed to be 64x64xchannels)
class CNNEncoder(nn.Module):
    def __init__(self, z_dim=3, c_dim=1, channels=1):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.pipe = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            Flatten3D(),
            nn.Linear(4*4*64, 256),
            nn.ReLU(True)
        )
        # map to shared
        self.S = nn.Linear(256, z_dim)
        # map to private
        self.P = nn.Linear(256, c_dim)

    def forward(self, x):
        tmp = self.pipe(x)
        shared = self.S(tmp)
        private = self.P(tmp)
        return shared, private


# CNNDecoder
class CNNDecoder(nn.Module):
    def __init__(self, z_dim=3, c_dim=1, channels=1):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.pipe = nn.Sequential(
            nn.Linear(z_dim+c_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 1024),
            nn.ReLU(True),
            Unflatten3D(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
        )

    def forward(self, s, p):
        # Reconstruct using both the shared and private
        recons = self.pipe(torch.cat((s, p), 1))
        return recons


# Multiview CNN Deterministic AutoEncoder
class CNNDAE(nn.Module):
    def __init__(self, z_dim=10, c_dim=2, channels=1):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_view = 2
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for _ in range(self.num_view):
            self.encoder.append(CNNEncoder(z_dim, c_dim, channels))
            self.decoder.append(CNNDecoder(z_dim, c_dim, channels))

    def encode(self, x):
        shared = []
        private = []
        for i in range(self.num_view):
            tmp = self.encoder[i](x[i])
            shared.append(tmp[0])
            private.append(tmp[1])

        return shared, private

    def decode(self, s, p):
        recons = []
        for i in range(self.num_view):
            tmp = self.decoder[i](s[i], p[i])
            recons.append(tmp)

        return recons

    def forward(self, x):
        shared, private = self.encode(x)
        recons = self.decode(shared, private)

        return shared, private, recons


