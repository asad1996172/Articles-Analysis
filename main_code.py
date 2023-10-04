import pandas as pd
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

SEED = 10


def clean_text(s: str) -> str:
    """
    Clean the given text string.
    :param s: Text string to be cleaned.
    :return: Cleaned text string.
    """
    s = s.lower()
    s = re.sub(r"\s\W|\\W,\s", " ", s)
    s = re.sub(r"[^\w]", " ", s)
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("co", "").replace("https", "").replace(",", "").replace("[\w*", " ")
    return s


def load_and_preprocess_data(filename: str) -> pd.DataFrame:
    """
    Load data from a CSV file and preprocess it.
    :param filename: Path to the CSV file.
    :return: Preprocessed data.
    """
    data = pd.read_csv(filename, encoding="ISO-8859-1")
    data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)
    data["Article"] = data["Article"].str.replace("strong>", "")
    data["Article"] = data["Article"].apply(clean_text)
    return data


def train_and_test_model(model, x_train, y_train, x_test, y_test):
    """
    Train a model and test its accuracy.
    :param model: The model to train and test.
    :param x_train: Training data.
    :param y_train: Training labels.
    :param x_test: Testing data.
    :param y_test: Testing labels.
    """
    model_name = type(model).__name__
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test) * 100
    print(f"<================ {model_name} =================>")
    print(f"Correct % for {model_name}: {accuracy:.2f} %")
    print("<=======================================================>\n")


if __name__ == "__main__":
    data = load_and_preprocess_data("Articles.csv")

    x = data["Article"]
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["NewsType"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    models = [
        KNeighborsClassifier(n_neighbors=3),
        MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(6,), random_state=SEED
        ),
        LogisticRegression(max_iter=10000, random_state=SEED),
        MultinomialNB(),
    ]

    for model in models:
        train_and_test_model(model, x_train, y_train, x_test, y_test)
