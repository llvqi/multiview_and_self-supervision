import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

def synthesize_samples(model, view1, view2, train_loader_b1, device):
    with torch.no_grad():
        model.eval()
        sample_dir = './synthesized_samples/'

        idx1 = 2182
        idx2 = 7080

        # sample used to generate fixed z_l (type)
        x_l = torch.tensor(view2[idx1]).float().to(device).unsqueeze(0)
        # sample used to generate fixed c_j (azimuth)
        x_j = torch.tensor(view2[idx2]).float().to(device).unsqueeze(0)

        # Get z_l
        z, _ = model.encode([x_l, x_l])
        z_l = z[1]

        # Get c_j
        _, c = model.encode([x_j, x_j])
        c_j = c[1]

        # Get a random batch
        train_iter = iter(train_loader_b1)
        x, y, _ = next(train_iter)
        x = x.to(device)
        y = y.to(device)

        # Generate nn samples
        nn = 64
        xx = x[-nn:]
        yy = y[-nn:]
        z, c = model.encode([xx, yy])

        # Generate samples with fixed z_l
        z1 = z_l.repeat(nn,1)
        zz = [z1, z1]
        syn_z = model.decode(zz, c)

        # Generate samples with fixed c_j
        c1 = c_j.repeat(nn,1)
        cc = [c1, c1]
        syn_c = model.decode(z, cc)


        # Save images
        # For each pair within a bounding box, the left data sample provides
        # c_j, and it is combined with the fixed z_l to get the synthetic
        # sample on the right.
        save_image(x_l, os.path.join(sample_dir, 'x_l.png'))
        x_concat = torch.cat([yy.view(-1, 3, 64, 64),
            syn_z[1].view(-1, 3, 64, 64)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'synthetic_fixed_z_l.png'))

        # For each pair within a bounding box, the left data sample provides
        # z_l, and it is combined with the fixed c_j to get the synthetic
        # sample on the right.
        save_image(x_j, os.path.join(sample_dir, 'x_j.png'))
        x_concat = torch.cat([yy.view(-1, 3, 64, 64),
            syn_c[1].view(-1, 3, 64, 64)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'synthetic_fixed_c_j.png'))
