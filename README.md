
# An Incremental Group-Specific Framework based on Community Detection ![exp](http://badges.github.io/stability-badges/dist/experimental.svg)

Paper is open resoursed at follow: https://ieeexplore.ieee.org/document/8795572
## Introduction
This is the project corresponding to the paper "Incremental Community Detection and
Group-Specific Model for Cold-Start Recommendation", the results and experiments are 
applied on computer with an Intel E5-2603 CPU. It is noticeable that the bipartite community detection
initialization method is run by an C++ application.

If you have any questions, please contact me by email: cs_xcy@126.com

*Besides, I am looking for a PhD position in 2020 fall. I would be really appreciated if you could provide or recommend any opportunities.*

## Environment
 |Package   |Version   |
| ------------ | ------------ |
|NumPy   |1.15.4   |
|Pandas   |0.23.4   |
|python-igraph   |0.7   |
|[biLouvain](https://github.com/paolapesantez/biLouvain)|-|

## Usage
We provide a python script to run a demo experiment.

    
    python script/demo.py
    
    
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

## Reference

> C. Xue, S. Wu, Q. Zhang and F. Shao, "An Incremental Group-Specific Framework Based on Community Detection for Cold Start Recommendation," in IEEE Access, vol. 7, pp. 112363-112374, 2019.
doi: 10.1109/ACCESS.2019.2935090
keywords: {recommender systems;social networking (online);incremental group-specific framework;cold start recommendation;cold start problem;rating information;recommender systems;decoupled normalization method;incremental community detection methods;incremental group-specific model;incremental data;Recommender systems;Collaboration;Complex networks;Data models;Computer science;Predictive models;Recommender systems;complex networks;incremental community detection;cold start},
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8795572&isnumber=8600701


