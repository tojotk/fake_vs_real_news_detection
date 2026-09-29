#  Fake vs Real News Detection

This project is about detecting whether a news article is **Fake or Real** using Natural Language Processing (NLP) and Machine Learning.

I worked on the project step by step, starting with understanding and cleaning the dataset, then preprocessing the text, training different machine learning models, analyzing the mistakes made by the models, and finally deploying the model using Streamlit.

---

##  About the Project

Fake news can spread very quickly through social media and online platforms. The aim of this project is to build a simple machine learning system that can identify patterns in news articles and classify them as:

* **0 → Fake News**
* **1 → Real News**

The model works based on the text patterns it learned from the training dataset. It is not a fact-checking system, so the prediction should not be considered proof that a real-world article is true or false.

---

##  Dataset

The dataset contains two files:

```text
True.csv
Fake.csv
```

I combined both datasets into one dataframe and added a label column.

```text
Fake News → 0
Real News → 1
```

Before training the models, I checked the dataset for missing values and duplicate records.

---

#  Task 1 — Text Exploration

The first step was to understand the dataset before building the model.

I performed:

* Basic dataset inspection
* Missing value checking
* Duplicate checking
* Class distribution analysis
* Article length analysis
* Vocabulary size analysis
* Word frequency analysis

I also visualized the class distribution and article length to get a better idea about the dataset.

### Some questions considered during EDA

**Do Fake and Real articles differ in length?**

The article length was calculated based on the number of words and compared between the two classes.

**What problems can noisy text create?**

News articles can contain URLs, punctuation, HTML tags, unnecessary spaces, and other text that may not be useful for classification. These can create extra features and make the model less efficient.

**Why is class balance important?**

If one class is much larger than the other, a model may become biased toward the majority class. Therefore, checking the class distribution is important before training.

---

#  Task 2 — NLP Preprocessing

After exploring the dataset, I cleaned the article text before using it for machine learning.

The preprocessing included:

* Converting text to lowercase
* Removing URLs
* Removing HTML tags
* Removing punctuation
* Removing extra spaces
* Handling English stop words

### TF-IDF

Since machine learning models cannot directly work with raw text, I used **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert the articles into numerical features.

I used:

```text
Unigrams + Bigrams
Maximum 50,000 features
English stop-word removal
```

The dataset was split into training and testing sets before fitting the TF-IDF vectorizer to avoid data leakage.

---

# 🤖 Task 3 — Machine Learning Models

I trained and compared three classification models:

### 1. Logistic Regression

Used as a simple and effective baseline for text classification.

### 2. Multinomial Naive Bayes

A commonly used algorithm for text classification problems.

### 3. Linear SVM

A linear Support Vector Machine that works well with high-dimensional TF-IDF text features.

---

##  Model Evaluation

I compared the models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The results were compared using a table and visualizations.

### Model Results

Add the actual results from the notebook here:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |        — |         — |      — |        — |       — |
| Naive Bayes         |        — |         — |      — |        — |       — |
| Linear SVM          |        — |         — |      — |        — |       — |

The final model was selected based on the evaluation results.

---

#  Task 4 — Error Analysis

After selecting the final model, I looked at the cases where the model made incorrect predictions.

I analyzed **15 misclassified articles** to understand why the model might have made mistakes.

Some possible reasons include:

* Similar writing styles between Fake and Real articles
* Ambiguous wording
* Short articles
* Unusual vocabulary
* Lack of enough contextual information

### Confusion Matrix

A confusion matrix was generated to see how well the final model classified Fake and Real news.

### Important Features

I also looked at the model coefficients to identify words and phrases that had a strong influence on the predictions.

This helped me understand what kind of textual patterns the model was learning.

---

#  Task 5 — Streamlit App

The final model was connected to a simple Streamlit application.

The application has three sections.

###  Home

The Home page gives a short overview of:

* The project
* Dataset
* Workflow
* Technologies used

###  Predict

Users can paste a news article and get:

* Fake / Real prediction
* Confidence score
* Fake probability
* Real probability

###  Explain

The Explain page provides information about:

* Dataset
* NLP preprocessing
* TF-IDF
* Machine learning model
* Important words/features used by the model

---

#  Technologies Used

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
TF-IDF
Natural Language Processing
Streamlit
Git & GitHub
```

---

#  Project Structure

```text
fake_vs_real_news_detection/
│
├── fake_vs_real_news_detection.ipynb
├── app.py
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

# How to Run

### 1. Clone the repository

```bash
git clone https://github.com/tojotk/fake_vs_real_news_detection.git
```

### 2. Open the project folder

```bash
cd fake_vs_real_news_detection
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The application will then open in the browser.

---

#  Limitations

This project has some limitations.

The model learns patterns from the dataset, so it may not perform equally well on every type of news article.

TF-IDF mainly focuses on word and phrase patterns and does not fully understand the meaning or context of an article.

Also, a high confidence score does not necessarily mean that the information in an article is factually correct.

---

#  Future Improvements

Some improvements I would like to explore in the future are:

* Using BERT or other Transformer models
* Using larger and more diverse datasets
* Adding multilingual news detection
* Adding SHAP or LIME for better explainability
* Connecting the system with external fact-checking sources
* Improving the Streamlit interface
* Deploying the application online

---


## ⭐ Project

This project was developed as part of my **Machine Learning / NLP practical work**, covering the complete process from data exploration to model deployment.
