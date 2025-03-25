from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_text(df):
    vectorizer = TfidfVectorizer(stop_words='english')  # load model
    return vectorizer.fit_transform(df['review/text'])  # text to embedings


def clustering(X):
    # Function to convert text into embedings
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)  # perform clustering
    return kmeans.fit_predict(X)


# Function to reduce dimensions for visualization
def dimensionality_reduction(X, clusters):
    svd = TruncatedSVD(n_components=2, random_state=42)  # reduce to 2 dimensions
    return svd.fit_transform(X)


# Function to plot clusters in a 2D space
def save_plot(reduced_data,clusters,file_path):
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
    plt.title("Cluster Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(file_path)
    plt.close()
