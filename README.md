# Introduction 
This project aims at extracting sensitive data, such as email ids and passwords, from Smart Reply generative language model using Service API and Model API attacks.
We consider two model architectures: decoder-only model that uses GPT2 pre-trained model checkpoint, and the encoder-decoder model that uses the Bert2Bert pre-trained model checkpoint.

# Installation
This project requires python3.8 (>= v3.8 is required by private-transformers). For executing some shell scripts, 'parallel' package is also required. Both these can be install using apt-get. Note that we also need to setup cuda packages (if not already present) in order to train models using GPUs on the local machine. Refer to https://medium.com/@anarmammadli/how-to-install-cuda-11-4-on-ubuntu-18-04-or-20-04-63f3dee2099 and https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73 for the instructions to setup cuda-11.x on ubuntu-18.04 machine.

Next, we would need to set up a virtual environment and then install all the dependencies using the following commands:
```
$ python3.8 -m venv env
$ source env/bin/activate
$ python3.8 -m pip install --upgrade pip
$ python3.8 -m pip install --no-cache-dir -r requirements.txt
```


# Description of Code Repository
The repository consists of the following python scripts:

- ```langauge_model_training.py``` contains the code for training / fine-tuning the models with canaries.
- ```langauge_model_inference.py``` contains the model evaluation and data extraction attack scripts.
- ```core_utilities.py``` contains the model architecture along with the utilities for model training. Also includes the scripts for data set pre-processing. 
- ```make_plot.py``` contains the code for plotting the loss and perplexity of model over the training and validation set, and for plotting the Rouge-2 scores over the validation set.