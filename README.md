
# An Incremental Group-Specific Framework based on Community Detection ![exp](http://badges.github.io/stability-badges/dist/experimental.svg)
## Introduction
This is the project corresponding to the paper "Incremental Community Detection and
Group-Specific Model for Cold-Start Recommendation", the results and experiments are 
applied on computer with an Intel E5-2603 CPU. It is noticeable that the bipartite community detection
initialization method is run by an C++ application.

## Environment
 |Package   |Version   |
| ------------ | ------------ |
|NumPy   |1.15.4   |
|Pandas   |0.23.4   |
|python-igraph   |0.7   |
|[biLouvain](https://github.com/paolapesantez/biLouvain)|-|

## Usage
We provide a python script to run all experiments at once.

    
    python script/test.py
    
Or you can run them one by one

    python script/mov_mis.py
    python script/mov_bipar.py
    ...
    
If you want to test on your own dataset, best to make sure the ID of users and items continuous and starts from 1. A good example of usage can be found in the any files located in scripts folder.
    
## Structure

|Folder  |Features   |
| ------------ | ------------ |
|out_groups   |The proposed incremental community detection methods.   |
|core   |Some central codes includes ALS and LSE algorithms.|
|data   |Extracted experimental data.|
|model   |A framework with the whole process.|
|scripts   |The entrance of experiments.|
|tools   |Some functions that are often used.|

    
    

