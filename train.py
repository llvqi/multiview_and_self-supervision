import torch
import torch.nn.functional as F
import numpy as np

import utils


# U update
def update_U(model, eval_loader, z_dim, device):
    model.eval()

    FF = []
    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(eval_loader):
            x = x.to(device)
            y = y.to(device)

            # Forward
            shared, _ = model.encode([x, y])
            FF.append(torch.cat(shared, 1))

        FF = torch.cat(FF, 0)

        # The projection step, i.e., subtract the mean
        FF = FF - torch.mean(FF, 0, True)

        h=[]
        for i in range(2):
            h.append(FF[:,i*z_dim:(i+1)*z_dim])

        FF = torch.stack(h, dim=2)

        # The SVD step
        S, _, T = torch.svd(torch.sum(FF, dim=2))
        U = torch.mm(S, T.t())
        U = U*(FF.shape[0])**0.5

    return U


# Compute correlation of x1 and x2
def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1*x2))/(sigma1*sigma2)

    return corr


# The loss function for matching and reconstruction
def loss_matching_recons(s, x_hat, x, U_batch):
    l = torch.nn.MSELoss(reduction='sum')

    # Matching loss
    match_err = l(torch.cat(s, 1), U_batch.repeat(1, 2))/s[0].shape[0]

    # Reconstruction loss
    recons_err = l(x_hat[0], x[0])
    recons_err += l(x_hat[1], x[1])
    recons_err /= s[0].shape[0]

    return match_err, recons_err


# The loss function for independence regularization
def loss_independence(phi_z1, tau_c1, phi_z2, tau_c2):
    # Correlation
    corr = compute_corr(phi_z1, tau_c1) + compute_corr(phi_z2, tau_c2)

    return corr


# Training function
def train(model, mmcca1, mmcca2, U, train_loader_b1, train_loader_b2,
        corr_iter, args, optimizer, device):

    model.train()
    mmcca1.train()
    mmcca2.train()

    for batch_idx, (x, y, idxes) in enumerate(train_loader_b1):
        x = x.to(device)
        y = y.to(device)

        # Forward with batch1
        shared, private, recons = model([x, y])

        # Get a batch2
        try:
            x_b2, y_b2, _ = next(corr_iter)
        except StopIteration:
            corr_iter = iter(train_loader_b2)
            x_b2, y_b2, _ = next(corr_iter)

        x_b2 = x_b2.to(device)
        y_b2 = y_b2.to(device)

        # Forward with batch2
        shared_b2, private_b2 = model.encode([x_b2, y_b2])


        # Using batch1
        match_err, recons_err = loss_matching_recons(shared, recons,
                [x, y], U[idxes, :])

        # Using batch2
        # Independence regularizer loss
        phi_z1, tau_c1 = mmcca1(shared_b2[0], private_b2[0])
        phi_z2, tau_c2 = mmcca2(shared_b2[1], private_b2[1])
        corr = loss_independence(phi_z1, tau_c1, phi_z2, tau_c2)

        # Compute the overall loss, note that we use the gradient reversal trick
        # and that's why we have a 'minus' for the last term
        loss = match_err + args.beta*recons_err - args._lambda*corr

        # Update all the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return corr_iter


# Evaluate on the training set
def eval_train(model, mmcca1, mmcca2, itr, U, eval_loader, args, device):
    model.eval()
    mmcca1.eval()
    mmcca2.eval()

    match_err = 0
    recons_err = 0

    # For indepnence computation over the wholse set
    s0,s1,p0,p1 = [],[],[],[]

    with torch.no_grad():
        for batch_idx, (x, y, idxes) in enumerate(eval_loader):
            x = x.to(device)
            y = y.to(device)

            # Forward
            shared, private, recons = model([x, y])
            s0.append(shared[0])
            s1.append(shared[1])
            p0.append(private[0])
            p1.append(private[1])

            # Matching and reconstruction loss
            m_e, r_e = loss_matching_recons(shared, recons, [x, y], U[idxes, :])

            match_err += m_e.item()*x.shape[0]
            recons_err += r_e.item()*x.shape[0]

        s0 = torch.cat(s0, 0)
        s1 = torch.cat(s1, 0)
        p0 = torch.cat(p0, 0)
        p1 = torch.cat(p1, 0)

        phi_z1, tau_c1 = mmcca1(s0, p0)
        phi_z2, tau_c2 = mmcca2(s1, p1)

        # Correlation over the whole set
        corr = loss_independence(phi_z1, tau_c1, phi_z2, tau_c2)


    match_err /= len(eval_loader.dataset)
    recons_err /= len(eval_loader.dataset)

    print('====> Iteration: {} total = {:.4f}, match = {:.4f}, recons = {:.4f}, corr = {:.7f}'.format(
        itr, match_err + args.beta*recons_err + args._lambda*corr.item(),
        match_err, recons_err, corr.item()))

    return match_err, recons_err, corr.item()
