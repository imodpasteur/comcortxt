# ComCorTxt

This project comprises two main scripts:

1. **bertopic_embedding**: This script allows to make bertopic embedding from a dataframe containing text field
2. **bertopic_plot_embedding**: This script allows to plot the embedding as shown in the paper

## Setup Instructions
Download the pretrained sentence camembert model:
- [Sentence CamemBERT Base](https://huggingface.co/dangvantuan/sentence-camembert-base)

To run the scripts, make a conda envirenment with python 3.9 and install the libraries as follow:

* conda create -n bert_env python=3.9
* conda activate bert_env
* pip install numpy scipy matplotlib umap-learn pandas hdbscan bertopic seaborn


## ⚠️ Warning: Dataset Not Provided

Please note that the dataset for which this code was written cannot be made publicly available without authorization by the French data protection authority Commission Nationale de l'Informatique et des Libertés (CNIL). You will need to replace the placeholder with your own dataset before running the code.
