# Project-A

This repository contains the results for the Rodinia Benchmark Test. 


# Creating Virtual Environment  

It is best to create a virtual environment first using 'virtualenv'. If you do not have it please install it using instructions in this link:

https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment  

Create a python virtual environment named venv2 using following command.  

python3 -m venv venv2  

Then activate the virtual environment using following command.  

source venv2/bin/activate

# To run the test do the following 

i. pip install -r requirements.txt  

ii. python3 main.py


It will show an output like this showing the summary of the results for the 15 loops in Rodinia Benchmark. It can be seen that all 15 loops are correctly predicted by the model. Also, there is a confusion matrix printed at the last which verfies the results in the table. It shows that all privates are detected as privates and all reductions are detected as reductions.


Num of correct predictions:  15  


|                   | precision|   recall | f1-score |  support|
|-------------------|----------|----------|----------|---------|
|  Private Clause   |     1.00 |    1.00  |   1.00   |    12   |
|  Reduction Clause |     1.00 |    1.00  |   1.00   |     3   |
|        accuracy   |          |          |   1.00   |    15   |
|       macro avg   |     1.00 |    1.00  |   1.00   |    15   | 
|    weighted avg   |     1.00 |    1.00  |   1.00   |    15   | 


Confusion Matrix  

|  		   |Private |  Reduction |
|------------------|--------|------------|
|  	Private	   |	12  |       0    |
|  	Reduction  |	0   |       3    |


# To train from scratch

i. pip install -r requirements.txt  

ii. python3 base.py

The previous one uses a pre-trained model to generate results on Rodinia. But this base.py will train the model from scratch and will generate the results for NAS Parallel Benchmark. dgl-csv-benchmark-test-2 contains the PerfoGraph representation of the loops. The base.py uses these loops for training the GNN model.
