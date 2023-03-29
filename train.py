import qiskit, nltk, time
import tensorflow_datasets as tfds
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.algorithms import QSVM
from qiskit.visualization import circuit_drawer, utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the IMDb movie review dataset
imdb_data, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

# Convert the dataset to a pandas DataFrame
imdb_df = pd.DataFrame(columns=["review", "sentiment"])
for review, sentiment in imdb_data["train"]:
    review = str(review.numpy().decode("utf-8"))
    sentiment = int(sentiment)
    imdb_df = imdb_df.append({"review": review, "sentiment": sentiment}, ignore_index=True)

# Preprocess the text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words and punctuation
    words = [word.lower() for word in words if word.lower() not in stop_words and word.isalnum()]

    # Lemmatize the remaining words
    words = [lemmatizer.lemmatize(word) for word in words]

    # Rejoin the words into a single string
    text = " ".join(words)

    return text

imdb_df["review"] = imdb_df["review"].apply(preprocess_text)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(imdb_df["review"], imdb_df["sentiment"], test_size=0.2, random_state=42)

# Create a bag-of-words representation of the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Define the QNN architecture
n_qubits = X_train.shape[1]
qc = QuantumCircuit(n_qubits)
start = time.time()
qc.append(RealAmplitudes(n_qubits, entanglement="linear", reps=1), range(n_qubits))
end = time.time()
print(end - start)
qc.measure_all()

fig = circuit_drawer(qc, output='mpl')
utils._save_figure_to_file('circuit.jpg', fig, 'jpg')  # or 'png' for PNG format


# Define the QSVM algorithm
optimizer = SPSA(max_trials=100)
qsvm = QSVM(qc, training_dataset=X_train, labels=y_train, test_dataset=X_test, optimizer=optimizer)

# Train the QSVM algorithm
backend = qiskit.Aer.get_backend("qasm_simulator")
result = qsvm.run(backend)

# Evaluate the QSVM algorithm on the test set
score = qsvm.test(X_test, y_test, backend=backend)
print("Test set accuracy:", score["testing_accuracy"])