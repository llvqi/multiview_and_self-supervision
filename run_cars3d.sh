#!/bin/bash

# Download data from http://www.scottreed.info
if [ ! -d "./data" ]
then
    echo "Downloading dataset ..."
    wget http://www.scottreed.info/files/nips2015-analogy-data.tar.gz

    echo "Decompress the data ..."
    tar -xzvf nips2015-analogy-data.tar.gz
fi

# Create sample directory
if [ ! -d "./synthesized_samples" ]
then
    mkdir "synthesized_samples"
fi

# Run the algorithm
python cars3d_demo.py \
--z_dim=10 \
--c_dim=2 \
--num_iters=200 \
--batchsize1=100 \
--batchsize2=800 \
--lr_max=1e0 \
--lr_min=1e-3 \
--weight_decay=1e-4 \
--beta=1e-1 \
--_lambda=1e0 \
--inner_epochs=10 \
--phi_num_layers=2 \
--phi_hidden_size=256 \
--tau_num_layers=2 \
--tau_hidden_size=256 \

