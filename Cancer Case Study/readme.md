
# Personalized Cancer Diagnosis

A brief description of what this Case Study Project and who it's for

## Context 

A lot has been said during the past several years about how precision medicine and, more concretely, how genetic testing is going to disrupt the way diseases like cancer are treated.

But this is only partially happening due to the huge amount of manual work still required. Memorial Sloan Kettering Cancer Center (MSKCC) launched this competition, accepted by the NIPS 2017 Competition Track,  because we need your help to take personalized medicine to its full potential.

Once sequenced, a cancer tumor can have thousands of genetic mutations. But the challenge is distinguishing the mutations that contribute to tumor growth (drivers) from the neutral mutations (passengers). 

Currently this interpretation of genetic mutations is being done manually. This is a very time-consuming task where a clinical pathologist has to manually review and classify every single genetic mutation based on evidence from text-based clinical literature.

## Business Problem Description

To develop algorithms to classify genetic mutations based on clinical evidence 

Source: https://www.kaggle.com/c/msk-redefining-cancer-treatment/

* Data: Memorial Sloan Kettering Cancer Center (MSKCC)

Downloading training_variants.zip and training_text.zip from Kaggle Source.

## Context

Source: https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/35336#198462

## Problem statement

Classify the given genetic variations/mutations based on evidence from text-based clinical literature.

## Source/Useful Links
Some articles and reference blogs about the problem statement

1. https://www.forbes.com/sites/matthewherper/2017/06/03/a-new-cancer-drug-helped-almost-everyone-who-took-it-almost-heres-what-it-teaches-us/#2a44ee2f6b25

2. https://www.youtube.com/watch?v=UwbuW7oK8rk

3. https://www.youtube.com/watch?v=qxXRKVompI8

## Real-world/Business objectives and constraints.
1. No low-latency requirement.

2. Interpretability is important.

3. Errors can be very costly.

4. Probability of a data-point belonging to each class is needed.

# Procedure followed to solve the business problem
 **_list of steps to be followed/performed_**:

1. Machine Learning Problem Formulation:
1.1 _Data_
    
1.2 _Data Overview_

1.3 _Eaxmple Data Point_

1.4 Understanding Columns
    
    Gene training_variants, training_text

2. Mapping the real-world problem to an ML problem

2.1 _Type of Machine Learning Problem_

2.2 _Performance Metrics_

2.3 _Machine Learning Objectives and constraints_

2.4 _Training, CV, Test Datasets_


3. Importing the required libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as pt
from scipy import stats
from wordcloud import STOPWORDS
from nltk import SnowballStemmer , PorterStemmer
from bs4 import BeautifulSoup
import re
import collections
import string
import math
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss , confusion_matrix
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB 
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
```
4. Exploratory Data Analysis 
4.1 _Univariate Analysis_
    
    Analysis of Gene Feature and ways to featurise this feature by using techniques like One-Hot Encoding, Mean-ResponseCoding, FeatureHashing and testing out how good is this feature in for our target var prediction by using a simple modelling techniques and documenting the observed results

    Analysis of Gene Variation Feature and ways to featurise this feature by using techniques like One-Hot Encoding, FeatureHasher, MeanResponseCoding and testing out how good is this feature in for our target var prediction by using a simple modelling techniques and documenting the observed results

    Analysis of Text Data and ways to featurise this data by using Bag of words, TFIDF and testing out how good is this feature in for our target var prediction by using a simple modelling techniques and documenting the observed results
4.2 _Multivariate Analysis_

Understanding the target variable behaviour by taking two features for modelling at a time and observing the results

4.3 _Modelling_
* _Creating Miscellenous Functions_
* _Loading the Datasets_ 
* _Applying Naive Bayes model_
* _Applying KNN Model_ 
* _Applying Logestic Regression Model_
* _Applying SVM Model_
* _Applying Rabdom Forest Model_
* _Applying a Stacking Classifier_

5. Dcoumenting the Results from above modelling
```
+---------------------+--------------------+------------------------+-----------------------------------------+----------------+-------------+--------------+
|     Feature Name    | Featurisation Used |         Model          |              HyperParameter             |  train Logloss |  Cv Logloss | test Logloss |
+---------------------+--------------------+------------------------+-----------------------------------------+----------------+-------------+--------------+
|         Text        |       TFIDF        |        Logistic        |                  0.001                  |     0.944      |     1.21    |    1.153     |
|         Text        |        BOW         |        Logistic        |                  0.001                  |     0.906      |    1.544    |    1.214     |
| gene,text,variation |      Hashing       |   StackingClassifier   | (0.1 Metaclassifier-LogisticRegression) |     0.313      |    0.526    |    0.526     |
| gene,text,variation |   OneHotEncoding   |   StackingClassifier   | (0.1 Metaclassifier-LogisticRegression) |      0.31      |    0.534    |    0.526     |
| gene,text,variation |   OneHotEncoding   |   StackingClassifier   |  (1 Metaclassifier-LogisticRegression)  |     0.309      |    0.531    |    0.527     |
| gene,text,variation | MeanResponseCoding |   StackingClassifier   | (0.1 Metaclassifier-LogisticRegression) |      0.14      |    0.242    |    0.259     |
| gene,text,variation |      Hashing       | RandomForestClassifier |        (estimatores-1000,depth-5)       |     0.884      |    1.125    |     1.14     |
| gene,text,variation | MeanResponseCoding | RandomForestClassifier |        (estimatores-2000,depth-3)       |     0.129      |    0.161    |    0.172     |
| gene,text,variation |   onehotencoding   | RandomForestClassifier |        (estimatores-2000,depth-5)       |      0.9       |    1.117    |    1.147     |
| gene,text,variation |      Hashing       |     SVM Classifier     |                    1                    |     0.972      |    1.133    |    1.203     |
| gene,text,variation | MeanResponseCoding |     SVM Classifier     |                    1                    |     0.878      |    1.104    |    1.057     |
| gene,text,variation | MeanResponseCoding |     SVM Classifier     |                   0.1                   |     0.936      |    1.151    |    1.168     |
| gene,text,variation | MeanResponseCoding |   LogisticRegression   |                    1                    |     0.873      |    1.192    |    1.199     |
| gene,text,variation |      Hashing       |   StackingClassifier   | (0.1 Metaclassifier-LogisticRegression) |     0.313      |    0.526    |    0.526     |
| gene,text,variation |   OneHotEncoding   |   StackingClassifier   | (0.1 Metaclassifier-LogisticRegression) |      0.31      |    0.534    |    0.526     |
| gene,text,variation |   OneHotEncoding   |   StackingClassifier   |  (1 Metaclassifier-LogisticRegression)  |     0.309      |    0.531    |    0.527     |
| gene,text,variation | MeanResponseCoding |   StackingClassifier   | (0.1 Metaclassifier-LogisticRegression) |      0.14      |    0.242    |    0.259     |
| gene,text,variation |      Hashing       | RandomForestClassifier |        (estimatores-1000,depth-5)       |     0.884      |    1.125    |     1.14     |
| gene,text,variation | MeanResponseCoding | RandomForestClassifier |        (estimatores-2000,depth-3)       |     0.129      |    0.161    |    0.172     |
| gene,text,variation |   onehotencoding   | RandomForestClassifier |        (estimatores-2000,depth-5)       |      0.9       |    1.117    |    1.147     |
| gene,text,variation |      Hashing       |     SVM Classifier     |                    1                    |     0.972      |    1.133    |    1.203     |
| gene,text,variation | MeanResponseCoding |     SVM Classifier     |                    1                    |     0.878      |    1.104    |    1.057     |
| gene,text,variation | MeanResponseCoding |     SVM Classifier     |                   0.1                   |     0.936      |    1.151    |    1.168     |
| gene,text,variation |       Hashed       |   LogisticRegression   |                    1                    |     0.873      |    1.192    |    1.199     |
| gene,text,variation | MeanResponseCoding |   LogisticRegression   |                    1                    |     0.584      |    0.857    |    0.821     |
| gene,text,variation |   onehotencoding   |   LogisticRegression   |                    1                    |     0.785      |    1.196    |    1.748     |
| gene,text,variation |       hashed       |     KNN Clasifier      |                    15                   |     0.647      |    0.751    |    0.736     |
| gene,text,variation | MeanResponseCoding |     KNN Clasifier      |                    31                   |     0.681      |    0.701    |    0.777     |
| gene,text,variation |   onehotencoding   |     KNN Clasifier      |                    31                   |     0.813      |    0.878    |    0.881     |
| gene,text,variation | MeanResponseCoding |      NB Clasifier      |                    1                    |     1.038      |    1.261    |    1.232     |
| gene,text,variation | MeanResponseCoding |      NB Clasifier      |                    1                    |     0.963      |    1.192    |    1.198     |
| gene,text,variation |      Hashing       |   StackingClassifier   | (0.1 Metaclassifier-LogisticRegression) |     0.313      |    0.526    |    0.526     |
| gene,text,variation |   OneHotEncoding   |   StackingClassifier   | (0.1 Metaclassifier-LogisticRegression) |      0.31      |    0.534    |    0.526     |
| gene,text,variation |   OneHotEncoding   |   StackingClassifier   |  (1 Metaclassifier-LogisticRegression)  |     0.309      |    0.531    |    0.527     |
| gene,text,variation | MeanResponseCoding |   StackingClassifier   | (0.1 Metaclassifier-LogisticRegression) |      0.14      |    0.242    |    0.259     |
| gene,text,variation |      Hashing       | RandomForestClassifier |        (estimatores-1000,depth-5)       |     0.884      |    1.125    |     1.14     |
| gene,text,variation | MeanResponseCoding | RandomForestClassifier |        (estimatores-2000,depth-3)       |     0.129      |    0.161    |    0.172     |
| gene,text,variation |   onehotencoding   | RandomForestClassifier |        (estimatores-2000,depth-5)       |      0.9       |    1.117    |    1.147     |
| gene,text,variation |      Hashing       |     SVM Classifier     |                    1                    |     0.972      |    1.133    |    1.203     |
| gene,text,variation | MeanResponseCoding |     SVM Classifier     |                    1                    |     0.878      |    1.104    |    1.057     |
| gene,text,variation | MeanResponseCoding |     SVM Classifier     |                   0.1                   |     0.936      |    1.151    |    1.168     |
| gene,text,variation |       Hashed       |   LogisticRegression   |                    1                    |     0.873      |    1.192    |    1.199     |
| gene,text,variation | MeanResponseCoding |   LogisticRegression   |                    1                    |     0.584      |    0.857    |    0.821     |
| gene,text,variation |   onehotencoding   |   LogisticRegression   |                    1                    |     0.785      |    1.196    |    1.748     |
| gene,text,variation |       hashed       |     KNN Clasifier      |                    15                   |     0.647      |    0.751    |    0.736     |
| gene,text,variation | MeanResponseCoding |     KNN Clasifier      |                    31                   |     0.681      |    0.701    |    0.777     |
| gene,text,variation |   onehotencoding   |     KNN Clasifier      |                    31                   |     0.813      |    0.878    |    0.881     |
| gene,text,variation | MeanResponseCoding |      NB Clasifier      |                    1                    |     1.038      |    1.261    |    1.232     |
| gene,text,variation | MeanResponseCoding |      NB Clasifier      |                    1                    |     0.963      |    1.192    |    1.198     |
+---------------------+--------------------+------------------------+-----------------------------------------+----------------+-------------+--------------+
```
