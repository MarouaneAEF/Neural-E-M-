#!/bin/bash
echo ##########################################################################
echo This process trains a neural version on Expectation-Maximization algorithm
echo ##########################################################################
# cloning repo of interest and dowloading paper's datasets to the data directory 
git clone --depth 1 https://github.com/MarouaneAEF/Neural-E-M-.git \
&& rm -rf Neural-E-M/.git/ \
&& cd Neural-E-M-/
wget -O ./data.zip https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AAB6WiZzH_mAtCjW6b9okMGea?dl=1

unzip data.zip -d data
chmod a+x data
rm data.zip 
# setting conda environment 
conda update conda
conda env create --name neural-stats --file environment.yml
conda activate neural-stats
# running experiments 
# python main.py 
# or 
python train_bernoulli.py
