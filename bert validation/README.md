# ComCorTxt

This project comprises two main scripts:

1. **preprocessing**: This script allows to split your dataset in training/validation/test sets and to extract tokens
2. **classification**: This script allows to train camembert on training set
3. **analysis**: This script allows to feed validation or test set to the trained network

## Setup Instructions

Download the required NLP models from Hugging Face:

- [CamemBERT Base](https://huggingface.co/almanach/camembert-base)
- [TF CamemBERT Base](https://huggingface.co/jplu/tf-camembert-base)

To run the scripts, make a conda envirenment with python 3.9 and install the libraries as follow:

* conda create -n camembert_env python=3.9.23
* conda activate camembert_env
* pip install numpy==1.24.3 scipy==1.13.1 matplotlib==3.9.2 pandas==2.2.3 bertopic==0.16.3 seaborn==0.13.2 tensorflow==2.8.0   transformers==4.26.1 protobuf==3.20.3



## ⚠️ Warning: Dataset Not Provided

Please note that the dataset for which this code was written cannot be made publicly available without authorization by the French data protection authority Commission Nationale de l'Informatique et des Libertés (CNIL). You will need to replace the placeholder with your own dataset before running the code.
