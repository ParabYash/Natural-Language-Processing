# Twitter Sentiment Analysis on the 2020 US Presidential Election

## Project Overview

This repository contains the project "Classification Model for Sentiment Analysis of 2020 US Election Tweets." The project investigates the correlation between social media trends during the 2020 US presidential election and public sentiment towards candidates Donald Trump and Joe Biden across the United States. By analyzing 4.1 million tweets, the project seeks to uncover insights into how political discourse on social media reflects public opinion and sentiment.

The project leverages **Natural Language Processing (NLP)** techniques for text cleaning, tokenization, and feature extraction, along with big data tools such as Apache Spark for distributed computing. Several machine learning models, including Logistic Regression, Random Forest, and Support Vector Machine (SVM), are used for sentiment classification.

## Contents

- **Research Paper**: [Twitter Sentiment Analysis Research Paper.pdf](./Twitter Sentiment Analysis Research Paper.pdf) – A comprehensive research report documenting the methodology, data preprocessing, model building, and findings of the project.
- **Jupyter Notebooks**:
  - `Twitter_Sentiment_Analysis.ipynb`: Initial data exploration, visualization, and pre-model analysis.
  - `Data_Preprocessing_and_Model_Building.ipynb`: Data preprocessing and model building, including machine learning model training and evaluation.
  
## Datasets

- **Twitter Data**: The dataset used consists of 4.1 million tweets related to the 2020 US presidential election. Due to its size, the dataset is not hosted in this repository but can be accessed from Kaggle [here](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets).

## Objectives

1. **Exploratory Data Analysis (EDA)**: To gain insights into the distribution of tweets, top countries, tweet sources, and the top five US states contributing to the conversation about each candidate.
2. **Natural Language Processing (NLP)**: Implement NLP techniques for cleaning, tokenizing, and preparing the tweet text for sentiment analysis.
3. **Data Preprocessing**: Clean and prepare the tweet data for sentiment analysis by filtering tweets, handling missing values, tokenizing text, and removing stopwords.
4. **Sentiment Classification**: Build and evaluate various machine learning models (Logistic Regression, Random Forest, and SVM) to classify tweets as positive or negative.
5. **Model Evaluation**: Compare model performance using metrics such as accuracy and select the best-performing model for sentiment classification.
6. **Sentiment Analysis Insights**: Analyze the distribution of sentiment across tweets related to Joe Biden and Donald Trump, providing insights into public opinion.

## Methodology

### 1. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of tweet counts across different countries.
   - Explore the top five US states contributing to tweet volumes for each candidate.
   - Analyze the distribution of tweet sources, such as web and mobile applications.

### 2. **Natural Language Processing (NLP)**:
   - **Text Cleaning**: Remove hashtags, usernames, URLs, emojis, and special characters from the text.
   - **Tokenization**: Split the cleaned text into individual words using RegexTokenizer in PySpark.
   - **Stopwords Removal**: Remove common stopwords (e.g., "the," "is") that carry little semantic meaning.
   - **HashingTF**: Convert the cleaned and tokenized text into numerical feature vectors using the hashing trick.

### 3. **Model Building**:
   - **Logistic Regression**: A supervised learning algorithm used for sentiment classification, transforming the textual data into probabilities of positive or negative sentiment.
   - **Random Forest Classifier**: An ensemble learning method that uses multiple decision trees to classify sentiment.
   - **Support Vector Machine (SVM)**: A powerful classification algorithm that finds the optimal hyperplane to separate positive and negative sentiments in the feature space.

### 4. **Model Evaluation**:
   - **Accuracy Comparison**: Evaluate model performance using accuracy scores.
   - **Best Model**: SVM was the best-performing model with an accuracy of 87.48%.

## Key Findings

- **Sentiment Analysis**: The SVM model revealed that Joe Biden received a higher portion of positive sentiment tweets than Donald Trump, which correlates with the overall election results.
- **Public Sentiment**: Donald Trump was the most talked-about candidate, but public sentiment leaned more positively towards Joe Biden.
- **Geographical Insights**: The majority of tweets came from the US, followed by the UK, India, and France. States such as California, New York, and Florida showed the highest tweet counts for both candidates.
- **Tweet Sources**: The majority of tweets were generated from the Twitter Web App, followed by mobile devices like iPhones and Androids, highlighting active participation in political discussions across platforms.

## Results Summary

- **Logistic Regression**: Accuracy of 85.34%
- **Random Forest**: Accuracy of 61.22%
- **SVM**: Accuracy of 87.48%

The Support Vector Machine (SVM) outperformed the other models, making it the best classifier for sentiment analysis in this project.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ParabYash/Twitter-Sentiment-Analysis.git
   ```

2. **Install Required Packages**:
   Ensure you have the necessary libraries installed, including PySpark:
   ```bash
   pip install pyspark
   ```

3. **Run the Notebooks**:
   - Open the Jupyter notebooks (`Twitter_Sentiment_Analysis.ipynb` and `Data_Preprocessing_and_Model_Building.ipynb`) in your preferred environment (JupyterLab, Jupyter Notebook, or any other IDE).
   - Execute the cells to replicate the data analysis, preprocessing, model building, and sentiment classification.

4. **Read the Report**:
   The research paper [Twitter Sentiment Analysis Research Paper.pdf](./Twitter Sentiment Analysis Research Paper.pdf) provides an in-depth explanation of the project, from the research questions and methodologies to the results and implications.

## Contact Information

For any questions or suggestions, feel free to contact me:

- **Email**: yashparab05@gmail.com
- **LinkedIn**: [Yash Parab](https://linkedin.com/in/yash-parab-9a5a6a209)
