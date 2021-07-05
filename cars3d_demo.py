import sys
import numpy as np

sys.path.append(".")
sys.path.append("..")

import torch
import model
import train
import utils
import evaluate

import argparse

'''
This is a demo for the NeurIPS 2021 submission 'Latent Correlation-Based
Multiview Learning and Self-Supervision: A Unifying Perspective'.
'''

# Argument parser
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--z_dim", default=10, help="Dimensionality of z", type=int)

    parser.add_argument("--c_dim", default= 2, help="Dimensionality of c", type=int)

    parser.add_argument("--num_iters", default=200,
            help="Number of training iterations", type=int)

    parser.add_argument("--batchsize1", default=100, help="Batch size for L and V", type=int)

    parser.add_argument("--batchsize2", default=800, help="Batch size for R", type=int)

    parser.add_argument("--lr_max", default=1e0,
            help="Learning rate for maximization", type=float)

    parser.add_argument("--lr_min", default=1e-3,
            help="Learning rate for minimization", type=float)

    parser.add_argument("--weight_decay", default=1e-4,
            help="Weight decay for parameters eta", type=float)

    parser.add_argument("--beta", default=1e-1,
            help="Reconstruction error coefficient", type=float)

    parser.add_argument("--_lambda", default=1e0, help="Regularizer coefficient", type=float)

    parser.add_argument("--inner_epochs", default=10, help="Number of inner epochs", type=int)

    # Structure for phi and tau network
    parser.add_argument("--phi_num_layers", default=2, help="Number of layers for phi", type=int)

    parser.add_argument("--phi_hidden_size", default=256, help="Number of hidden neurons for phi",
            type=int)

    parser.add_argument("--tau_num_layers", default=2, help="Number of layers for tau", type=int)

    parser.add_argument("--tau_hidden_size", default=256, help="Number of hidden neurons for tau",
            type=int)

    return parser


def main(args):
    parser = get_parser()
    args = parser.parse_args(args)

    torch.manual_seed(0)
    np.random.seed(12)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Encoder and decoder network
    ae_model = model.CNNDAE(args.z_dim, args.c_dim, channels=3).to(device)
    # View1 independence regularization network
    mmcca1 = model.MMDCCA(args.z_dim, args.c_dim,
            [args.phi_hidden_size]*args.phi_num_layers,
            [args.tau_hidden_size]*args.tau_num_layers).to(device)
    # View2 independence regularization network
    mmcca2 = model.MMDCCA(args.z_dim, args.c_dim,
            [args.phi_hidden_size]*args.phi_num_layers,
            [args.tau_hidden_size]*args.tau_num_layers).to(device)


    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': mmcca1.parameters(), 'lr': args.lr_max, 'weight_decay': args.weight_decay},
        {'params': mmcca2.parameters(), 'lr': args.lr_max, 'weight_decay': args.weight_decay},
        {'params': ae_model.parameters(), 'lr': args.lr_min}
        ], lr=args.lr_min)


    # Construct data loaders
    print("Preparing data ...")
    view1, view2 = utils.get_cars3d()

    # Shffule the correspondence of the 'azimuth' (private) information
    shffled_idx = utils.sample()

    train_loader_b1 = utils.get_dataloader(view1, view2[shffled_idx], args.batchsize1, True)
    eval_loader = utils.get_dataloader(view1, view2[shffled_idx], args.batchsize2, False)
    train_loader_b2 = utils.get_dataloader(view1, view2[shffled_idx], args.batchsize2, True)
    # Batch iterator for the independence regularizer
    corr_iter = iter(train_loader_b2)


    # Start training
    best_obj = float('inf')
    model_file_name = 'cars3d_model.pth'

    print("Start training ...")
    for itr in range(1, args.num_iters+1):

        # Solve the U subproblem
        U = train.update_U(ae_model, eval_loader, args.z_dim, device)

        # Update network theta and eta for multiple epochs
        for _ in range(args.inner_epochs):

            # Backprop to update
            corr_iter = train.train(ae_model, mmcca1, mmcca2, U, train_loader_b1,
                    train_loader_b2, corr_iter, args, optimizer, device)

            # Evaluate on the whole set
            match_err, recons_err, corr = train.eval_train(ae_model, mmcca1, mmcca2,
                    itr, U, eval_loader, args, device)

            # Save the model
            if match_err + args.beta*recons_err + args._lambda*corr < best_obj:
                print('Saving Model')
                torch.save(ae_model.state_dict(), model_file_name)
                best_obj = match_err + args.beta*recons_err + args._lambda*corr


    # Load model
    ae_model.load_state_dict(torch.load(model_file_name))
    ae_model = ae_model.to(device)

    # Synthesize samples
    print("Synthesize samples ...")
    evaluate.synthesize_samples(ae_model, view1, view2, train_loader_b1, device)


if __name__ == "__main__":
    main(sys.argv[1:])

