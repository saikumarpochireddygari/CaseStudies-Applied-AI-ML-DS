
#  Quora Question Pairs Similarity

A brief description of what this project is

## Context
Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

credits: kaggle

## Problem Statement
* Identify which questions asked on Quora are duplicates of questions that have already been asked.
* This could be useful to instantly provide answers to questions that have already been answered.
* We are tasked with predicting whether a pair of questions are duplicates or not.

## Realworld Business
* The cost of a mis-classification can be very high.
* You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
* No strict latency concerns.
* Interpretability is partially important.

## Procedure followed to solve the business problem

### 1. _Data:_ 

- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate
- Size of Train.csv - 60MB
- Number of rows in Train.csv = 404,290

    * _Example Data Point:_

    ```python
    "id","qid1","qid2","question1","question2","is_duplicate"
    "0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
    "1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
    "7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"
    "11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"
    ```
### 2. _Mapping the real world problem to an ML problem:_
*  Type of Machine Leaning Problem
    > It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.

### 3. _Performance Metrics:_
* log-loss 
* Binary Confusion Matrix

### 4. _Train and Test Data Construction:_
We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.

### 5. _Exploratory Data Analysis:_ 
* importing required libraries
    ```python
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly 
    import math
    import string
    import re
    import nltk
    from nltk import SnowballStemmer , PorterStemmer
    import collections 
    from bs4 import BeautifulSoup
    from wordcloud import STOPWORDS
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss , confusion_matrix 
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBClassifier

    ```
* Understanding few questions about data: 
    * How is the class label ( is_duplicate ) distributed with respect to data points?
    * Are these questions repeating multiple times?
    * Can we see unique questions and repeated questions ?
* Fearisation to get more insights about the data that help in objective of classification
    
    As our data set is having question1 and question2 features just by looking at these we cannot make sense as we cannot plot them as they are actual questions itself and by logic we know that if two questions are different then there will/will not be different/not different words with or without the semantic meanings of the words everything depends on the context. As we are humans reading the pair of questions it will be easy to understand for us and differentiate .For a machine to differentiate means it needs data in machine readable form that is numbers. Here in this part we will create some own features based on the questions we have with out cleaning the questions and preproccesing them and perform EDA on them ,Later we can convert sentances and create advance features and do EDA on them as well to know these features are helpful or not.

    * no_words_in_question1 :- total words in question1
    * no_words_in_question2 :- total words in question2
    * len_of_question1 :- length of the question1
    * len_of_question2 :- length of the question2
    * unique_commonwords_inboth_qestions :- total common words which are unique to both questions

    * frequency_of_question1 :- no of times this question1 occurs

    * frequency_of_question2 :- no of times this question2 occurs
    * word_share :- this is basically words shared between two sentances,uniquecmmnwords q1+q2/totalnoofwordsin q1+q2
    * freq1+freq2 :- freqency of q1 + freq q2
    * freq1-freq2 :- abs(frequency of q1 - freq q2)
    * total_noof_words_q1+q2 :- no of words in question1+question2
    * Advaced Features:- 
        * cwc_min : Ratio of common_word_count to min lenghth of word count of Q1 and Q2
        * cwc_min = common_word_count / (min(len(q1_words), len(q2_words))

        * cwc_max : Ratio of common_word_count to max lenghth of word count of Q1 and Q2
        * cwc_max = common_word_count / (max(len(q1_words), len(q2_words))

        * csc_min : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2
        * csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))

        * csc_max : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2
        * csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))

        * ctc_min : Ratio of common_token_count to min lenghth of token count of Q1 and Q2
        * ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))


        * ctc_max : Ratio of common_token_count to max lenghth of token count of Q1 and Q2
        * ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))


        * last_word_eq : Check if First word of both questions is equal or not
        * last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])


        * first_word_eq : Check if First word of both questions is equal or not
        * first_word_eq = int(q1_tokens[0] == q2_tokens[0])


        * abs_len_diff : Abs. length difference
        * abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))


        * mean_len : Average Token Length of both Questions
        * mean_len = (len(q1_tokens) + len(q2_tokens))/2


        * fuzz_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/


        * fuzz_partial_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/


        * token_sort_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

        * token_set_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

* Univariate Analysis 
* BiVariable Analysis
Documenting the results from the analysis.
### 6. _Data Cleaning_
### 7. _Featurization_
### 8. _ Modelling:_
* _Applying Linear SVM algorithm_
* _Logistic Regression algorithm_

### 9. _Documenting the results:_
```

+------------+--------------------+-----------------+---------+
| Vectorizer |  classifier used   | Hyper Parameter | LogLoss |
+------------+--------------------+-----------------+---------+
|   array    |    random Model    |       null      |    13   |
|   TFIDF    | LogisticRegression |       0.01      |  0.4286 |
|   TFIDF    |     Linear SVM     |       0.01      |  0.4318 |
+------------+--------------------+-----------------+---------+
We can notice logistic regresion performed better than all we can infer from the result table.Linear SVM also performed Good.
```
