import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data(df):
    # Convert the "review/score" column to numeric value
    df["review/score"] = pd.to_numeric(df["review/score"], errors="coerce")

    # Create a new column to classify reviews as positive or negative
    df["semantic"] = df["review/score"].apply(lambda x: "positive" if x >= 4 else "negative")

    # Combine the "review/summary" and "review/text" columns into one column
    df["review_combined"] = df["review/summary"] + " " + df["review/text"]


def vectorize_text(df):
    tfidf_vectorizer = TfidfVectorizer()
    return tfidf_vectorizer.fit_transform(df['review_combined'])  # text to embedings


def train_and_evaluate_model(X_train, y_train, X_test, y_test, file_path):
    model = LogisticRegression()  # load model
    model.fit(X_train, y_train)  # train model

    y_pred = model.predict(X_test)  # prediction

    # check results
    report = classification_report(y_test, y_pred)

    # save resul to file
    with open(file_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(report)


def save_plot(df,sentimental_photo_path):
    counts = df["semantic"].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, palette=["red", "green"])
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Distribution of Positive and Negative Reviews")
    plt.savefig(sentimental_photo_path)
    plt.close()
