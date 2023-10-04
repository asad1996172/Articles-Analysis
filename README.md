# Articles Analysis
Predictive analysis of a dataset containing news articles from 2015 to 2018, related to business and sports, sourced from Kaggle. The primary objective of this project was to compare the performance of various machine learning methods using a simple bag-of-words approach for classification.

Features:
- Word Analysis: Provides an analysis of the most frequent words in the dataset for both news types (Business and Sports).
- Predictive Analysis: Compared the performance of several machine learning models, including K Nearest Neighbors (KNN), Logistic Regression (LR), Forward-Feed Neural Networks (FNN), and Multinomial Naive Bayes (MNB), on the task of predicting the type of news article (Business or Sports) based on its content.

Technical Details:
- Programming Language: Python
- Data Cleaning: Regular expressions and pandas
- Machine learning: Scikit-learn
- Visualization: Matplotlib and Seaborn

Kaggle Dataset: https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles

### Instructions to Run Code
## Demo
[![Video Instructions to Run Code](https://i.ytimg.com/vi/qnMmEbggDm0/hqdefault.jpg)](https://www.youtube.com/watch?v=qnMmEbggDm0)

This code requires python3.6. 
1) Clone the repository: `git clone https://github.com/asad1996172/Articles-Analysis`
2) Create a conda environment: `conda create -n articles python=3.6`
3) Activate the environment: `conda activate articles`
4) Install the requirements:`pip install -r requirements.txt`
5) Run the code: `python3 main_code.py`

### Discussion and Results
First, I did some word analysis on the different News Types. There are two News Types in this dataset i.e., Business and Sports. Following are the top 20 words with respect to frequency along with their number of occurences.

![Business Words](business_words.jpg) 

![Sports Words](sports_words.jpg) 

Intersection of these two sets contains very less words so intuitively classification accuracies are expected to be high.
Following are the models we use for comparison 
1) K Nearest Neighbors [(KNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
2) Logistic Regression [(LR)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
3) Forward-Feed Neural Networks [(FNN)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
4) Multinomial Naive Bayes [(MNB)](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

I used the 80-20 train test split. Following are the accuracies for different models on the 20% test set.

| Classifier | Accuracy  |
|------------|---------- |
| KNN (k=3)  | 93.88%    |
| LR         | 99.26%    |
| FNN        | 99.07%    |
| MNB        | **99.63%**|

As evident from the table, the best performing model among these is Multinomial Naive Bayes.
