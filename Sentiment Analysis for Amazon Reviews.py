# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
# Business Problem
# Kozmos, a company focusing on home textiles and casual clothing products sold through Amazon, aims to increase its sales
# by analyzing customer reviews and improving its products based on the complaints received. To achieve this goal,
# sentiment analysis will be performed on the reviews, and the labeled data will be used to build a classification model.


# Dataset Story
# The dataset consists of variables related to reviews of a specific product group, including the review title,
# star rating, and the number of people who found the review helpful.

# Review: The review made on the product
# Title: The title given to the review content, a short comment
# HelpFul: The number of people who found the review helpful
# Star: The number of stars given to the product

# Tasks

# Task 1

# Task 1: Text Preprocessing Steps
# 1. Read the `amazon.xlsx` data.
# 2. On the "Review" variable:
   # - a. Convert all letters to lowercase.
   # - b. Remove punctuation marks.
   # - c. Remove numerical expressions found in the reviews.
   # - d. Remove non-informative words (stopwords) from the data.
   # - e. Remove words that appear less than 1000 times in the data.
   # - f. Apply lemmatization.

# Task 2: Text Visualization
    # Step 1: Barplot Visualization**
        #    - a. Calculate the word frequencies contained in the "Review" variable and save them as `tf`.
        #    - b. Rename the columns of the `tf` dataframe as "words" and "tf".
        #    - c. Filter the data to include only those with a "tf" value greater than 500 and complete the visualization using a barplot.

    # Step 2: Word Cloud Visualization
        # a. Save all words from the "Review" variable as a string named "text".
        # b. Create a WordCloud template and save it.
        # c. Generate the WordCloud using the template and the string created in the previous step.
        # d. Complete the visualization steps (figure, imshow, axis, show).


# Task 3: Sentiment Analysis
    # Step 1: Create a SentimentIntensityAnalyzer object using NLTK
    # Step 2: Analyze sentiment scores using SentimentIntensityAnalyzer
        # a. Calculate polarity_scores() for the first 10 observations in the "Review" variable.
        # b. Filter and review the first 10 observations based on their compound scores.
        # c. Update the observations: if compound scores are greater than 0, label as "pos"; otherwise, label as "neg".
        # d. Create a new variable in the dataframe with pos-neg assignments for all observations in the "Review" variable.
        # NOTE: By labeling the comments with SentimentIntensityAnalyzer, the dependent variable for the text classification machine learning model is created.



# Task 4: Preparing for Machine Learning!
    # Step 1: Determine the dependent and independent variables and split the data into train and test sets.
    # Step 2: Convert the representations to numerical format for machine learning model input.
        # a. Create an object using TfidfVectorizer.
        # b. Fit the object using the previously split train data.
        # c. Apply transformation to the train and test data using the created vector and save.



# Task 5: Modeling (Logistic Regression)
    # Step 1: Build and fit the logistic regression model using the train data.
    # Step 2: Make predictions with the created model.
        # a. Predict the test data and save the results.
        # b. Report and review the prediction results using classification_report.
        # c. Calculate the average accuracy using cross-validation.
    # Step 3: Ask the model about randomly selected comments from the data.
        # a. Use the sample function to select samples from the "Review" variable and assign to a new variable.
        # b. Vectorize the sample using CountVectorizer for the model to predict.
        # c. Fit and transform the vectorized sample and save it.
        # d. Predict the sample using the created model and save the results.
        # e. Print the sample and the prediction result.



# Task 6: Modeling (Random Forest)
    # Step 1: Build and fit a Random Forest model and review the prediction results.
        # a. Create and fit a RandomForestClassifier model.
        # b. Calculate the average accuracy using cross-validation.
        # c. Compare the results with the logistic regression model.



#Step 1
import pandas as pd
df = pd.read_excel("C:\\Users\\hazal\\OneDrive\\Masaüstü\\datasets\\amazon.xlsx")
df.head()
df.info()

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.width",200)
pd.set_option("display.float_format",lambda x: "%.2f" % x)

# Step 2
# a)
df["Review"]=df["Review"].str.lower()
# b
df["Review"] = df["Review"].str.replace("[^\w\s]","")
# c
df["Review"] = df["Review"].str.replace("/d","")

# d)
import nltk
nltk.download("stopwords")
sw = stopwords.words("english")
df["Review"] =df["Review"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw)

# e
df["Review"] = df["Review"].fillna("")
sil = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))


# f
import nltk
nltk.download("wordnet")
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Görev 2
# Adım 1
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words","tf"]
tf[tf["tf"] > 500].plot.bar(x="words",y="tf")
plt.show()

# Adım 2
text = " ".join(i for i in df.Review)
wordcloud = WordCloud(max_font_size =50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Görev 3

# adım 1
sia = SentimentIntensityAnalyzer()

# adım 2
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["Sentiment_Label"] = df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df.groupby("Sentiment_Label")["Star"].mean()

# Görev 4

train_x, test_x, train_y, test_y =train_test_split(df["Review"],df["Sentiment_Label"],random_state=42)

tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# Görev 5
# adım 1
logistic_regression = LogisticRegression().fit(x_train_tf_idf_word,train_y)

# adım 2
y_pred = logistic_regression.predict(x_train_tf_idf_word)
print(classification_report(y_pred,test_y))

cross_val_score(logistic_regression,x_test_tf_idf_word,test_y,cv=5).mean()


# Görev 6

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()





















