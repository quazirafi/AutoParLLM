# AutoParLLM

This repository contains the code and dataset for the NAACL 2025 main conference paper "AutoParLLM: GNN-guided Context Generation for Zero-Shot Code
Parallelization using LLMs". 


# Creating Virtual Environment  

It is best to create a virtual environment first using 'virtualenv'. If you do not have it please install it using instructions in this link:

https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment  

Create a python virtual environment named venv2 using following command.  

python3 -m venv venv2  

Then activate the virtual environment using following command.  

source venv2/bin/activate

# To run the test on the Rodinia Benchmark do the following 

i. pip install -r requirements.txt  

ii. python3 main.py


# To train from scratch

i. pip install -r requirements.txt  

ii. python3 base.py

The previous one uses a pre-trained model to generate results on Rodinia. But this base.py will train the model from scratch and will generate the results for NAS Parallel Benchmark. dgl-csv-benchmark-test-2 contains the PerfoGraph representation of the loops. The base.py uses these loops for training the GNN model.
