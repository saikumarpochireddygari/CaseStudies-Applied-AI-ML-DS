
 
# Facebook Friend Reccomendation 

A brief description of what this Case Study Project and who it's for


# Procedure followed to solve the business problem


### 1. _Problem Statement_

Given a directed social graph, have to predict missing links to recommend users (Link Prediction in graph)

## 2. _Data Overview_
Taken data from facebook's recruting challenge on kaggle https://www.kaggle.com/c/FacebookRecruiting
data contains two columns source and destination eac edge in graph

```
- Data columns (total 2 columns):  
- source_node         int64  
- destination_node    int64  

```

## 3. _Mapping the problem into supervised learning problem:_
Generated training samples of good and bad links from given directed graph and for each link got some features like no of followers, is he followed back, page rank, katz score, adar index, some svd fetures of adj matrix, some weight features etc. and trained ml model based on these features to predict link.
Some referred papers and videos :
* https://www.cs.cornell.edu/home/kleinber/link-pred.pdf
* https://www3.nd.edu/~dial/publications/lichtenwalter2010new.pdf
* https://kaggle2.blob.core.windows.net/forum-message-attachments/2594/supervised_link_prediction.pdf
* https://www.youtube.com/watch?v=2M77Hgy17cg

## 4. _Business objectives and constraints:_¶
* No low-latency requirement.
* Probability of prediction is useful to recommend highest probability links

## 5. _Performance Metrics:_

Performance metric for supervised learning:
* Both precision and recall is important so F1 score is good choice
* Confusion matrix

## 6. _Importing Libraries:_

```python
import warnings
warnings.filterwarnings("ignore")
import csv
import pandas as pd
import datetime 
import time
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.cluster import MiniBatchKMeans, KMeans
import math
import pickle
import os
import xgboost as xgb
import warnings
import networkx as nx
import pdb
import pickle
from sklearn.model_selection import GridSearchCV
```
## 7. _Exploratory Data Analysis:_
###### Exploring few basic questions 
* No of followers for each person
* No of people each person is following
* both followers + following
## 8. _Understanding Graphs:_
* Generating some edges which are not present in graph for supervised learning
## 9. _Feature Engineering_
* Similarity measures
    * Jaccard Distance
    * Cosine distance
* Ranking Measures
    * Page Ranking
* Other Graph Features
    * Shortest path
    * Adamic/Adar Index
    * Adamic/Adar Index
    * Katz Centrality
    * Hits Score
Adding all these features to the data sets.

## 8. _ Adding a set of features:_
* jaccard_followers
* jaccard_followees
* cosine_followers
* cosine_followees
* num_followers_s
* num_followees_s
* num_followers_d
* num_followees_d
* inter_followers
* inter_followees
* adar index
* is following back
* belongs to same weakly connect components
* shortest path between source and destination
* Weight Features:-
    * weight of incoming edges
    * weight of outgoing edges
    * weight of incoming edges + weight of outgoing edges
    * weight of incoming edges * weight of outgoing edges
    * 2*weight of incoming edges + weight of outgoing edges
    * weight of incoming edges + 2*weight of outgoing edges
* Page Ranking of source
* Page Ranking of dest
* katz of source
* katz of dest
* hubs of source
* hubs of dest
* authorities_s of source
* authorities_s of dest
* Adding new feature Preferential attachment :-
    * Preferential Attachment One well-known concept in social networks is that users with many friends tend to create more connections in the future. This is due to the fact that in some social networks, like in finance, the rich get richer. We estimate how ”rich” our two vertices are by calculating the **multiplication between the number of friends (|Γ(x)|) or followers each vertex has. It may be noted that the similarity index does not require any node neighbor information; therefore, this similarity index has the lowest computational complexity.
* svd_dot:-
    * you can calculate svd_dot as Dot product between sourse node svd and destination node svd features.

Adding all these features to the data sets.

# 9. _Modelling:_

* _Applying Random forest model_
* _Applying XGBoost Model¶_

# 10. _Observing the Key Metrics_
Observed the key performance metrics for the problem Statement.
