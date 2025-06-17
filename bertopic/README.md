# ComCorTxt

This project comprises two main scripts:

1. **bertopic_embedding**: This script allows to make bertopic embedding from a dataframe containing text field
2. **bertopic_plot_embedding**: This script allows to plot the embedding as shown in the paper

## Setup Instructions
Download the pretrained sentence camembert model:
- [Sentence CamemBERT Base](https://huggingface.co/dangvantuan/sentence-camembert-base)

To run the scripts, make a conda envirenment with python 3.9 and install the libraries as follow:

* conda create -n bertopic_env python=3.9.23
* conda activate bertopic_env
* pip install numpy==1.24.3 scipy==1.13.1 matplotlib==3.9.2 pandas==2.2.3 bertopic==0.16.3 seaborn==0.13.2 sentence-transformers==3.1.1



## ⚠️ Warning: Dataset Not Provided

Please note that the dataset for which this code was written cannot be made publicly available without authorization by the French data protection authority Commission Nationale de l'Informatique et des Libertés (CNIL). You will need to replace the placeholder with your own dataset before running the code.
