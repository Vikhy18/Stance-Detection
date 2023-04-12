# Stance Detection

In this project, we aim to tackle one of the prominent problems in Natural Language Processing that is Stance Detection. We use the tweets as the source of information to predict the stance against a target topic.

## 📚 Dataset

We have used SemEval6A that comprises of 2814 tweets related to the below listed topics along with the stance of each tweet pertaining to the target.

1. *Atheism*
2. *Legalization of Abortion*
3. *Climate change is a real concern*
4. *Hillary Clinton*
5. *Feminist Movement*

We have also used *tweepy* to scrape tweets from twitter for the following topics:

1. *Gun Laws*
2. *Immigration*
3. *Racism*

## 🚀 Getting Started

To get started with this project, follow these steps:

1. Make sure you have git installed on your local machine.
2. Clone this repository onto your local machine.  
```git pull https://github.com/Vikhy18/Stance-Detection.git```
3. Install the Python package `pip` to install the dependencies.
4. Install all the requirements using `pip`.  
```pip install -r requirements.txt```
5. Start the dash web-application using the following command.  
```python webapp.py```
6. Web-application should be launched at *https://127.0.0.1/8050*

## 💻 Algorithms

We had labelled dataset from SemEval6A. But, for the tweets scraped from twitter we had to assign the appropriate labels. For this, we used K-Means algorithm with TF-IDF vectorization to cluster the tweets into three classes. Then we analyzed the some samples from each class and assigned the appropriate class to each cluster.

We used the following preprocessing and data augmentation techniques:


**Data Preprocessing**
* Data Cleaning - Removing URLs and mentions,  replacing emoticons with their corresponding sentiment, correcting any spelling errors in the text.
* Tokenization -  Splitting the text into individual words to enable us to represent the text as numerical vectors.
* Removal of stop words - Removing common words (like and, the, a, an, etc.) to focus more on the important and informative parts of the text. 
* Stemming and Lemmatization - Clustering similar words together for pinpointing pertinent keywords and extracting the most significant features. 


**Data Augmentation**
* Data Expansion - Scraping Twitter for more tweets related to the existing topics plus recently trending topics. 
* Synonym expansion - Including synonyms to capture diverse opinions understand more nuanced contexts.
* Phrase expansion - Expanding the use of related phrases to improve the comprehensiveness of the analysis.
* Query reformulation - By reformulating the query, the model can become more adaptable to predict the stance regardless of the specific way in which an opinion is expressed.


We used BERT with SVM and Bi-LSTMs for predicting the stance.

**BERT with SVM**
* With its ability to analyze word and phrase relationships in a sentence, BERT is an ideal tool for capturing the context and meaning of the text. 
* We used pre-trained BERT-Base model to generate a set of features, which were then used as input to an SVM classifier for predicting stance. SVM is a suitable choice because it can establish clear decision boundaries for the three stances (Favor, Against, and None).

**Bidirectional Long Short-Term Memory Networks (Bi-LSTMs)**
* Bi-LSTMs are suitable because of their ability to process sequential text in both forward and backward directions, allowing them to capture contextual information from the entire input sequence.
* Furthermore, Bi-LSTMs can overcome the issue of vanishing gradients in deep neural networks by utilizing LSTM cells, which maintain information over time.


## 👥 Team Members

- Avish Khosla
- Baibhav Phukan
- Gautham Maraswami
- Sai Rathnam Pallayam Ramanarasaiah 
- Sai Vikhyath Kudhroli
- Tanuja Renu Sudha
