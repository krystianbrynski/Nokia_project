import click
import yaml
import gzip
import pandas as pd
from sklearn.model_selection import train_test_split
from Sentimental_Analysis import sentimental
from Clustering import clustering
from Product_Recommendation import recommendation


def parse(filename):
    f = gzip.open(filename, 'rt')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = l.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        eName = l[:colonPos]
        rest = l[colonPos + 2:]
        entry[eName] = rest
    yield entry


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run_pipeline(config) -> None:
    with open(config, 'r') as file:
        config_data = yaml.safe_load(file)
        file_path = config_data.get('file_path')  # load path for data
        sentimental_scores_path = config_data.get('sentimental_scores_path')
        sentimental_photo_path = config_data.get('sentimental_photo_path')
        cluster_photo_path = config_data.get('cluster_photo_path')
        similar_products_path = config_data.get('similar_products_path')
        input_product_id = config_data.get('input_product_id')

        records = [e for e in parse(file_path)]  # parse doc
        df = pd.DataFrame(records)  # create data frame
        df = df.dropna()  # drop Nan rows

        # Create 3 datasets for each purpose: Sentiment Analysis, Clustering and Recommendation System
        df_sentimental = df
        df_clustering = df
        df_recommendation = df

        # Sentimental Analysis
        sentimental.prepare_data(df_sentimental)  # prepare data for sentimental analysis
        X_tfidf = sentimental.vectorize_text(df_sentimental)  # creating embeddings (text to number)

        # split data for test and train dataset
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df_sentimental["semantic"], test_size=0.2,
                                                            random_state=42)

        sentimental.train_and_evaluate_model(X_train, y_train, X_test, y_test,
                                             sentimental_scores_path)  # trian and test model
        sentimental.save_plot(df_sentimental, sentimental_photo_path)

        # Clustering
        X = clustering.vectorize_text(df_clustering)  # creating embeddings (text to number)
        clusters = clustering.clustering(X)  # Clustering
        reduced_data = clustering.dimensionality_reduction(X, clusters)  # Reduce dimension
        clustering.save_plot(reduced_data, clusters, cluster_photo_path)  # visualisation

        # Recommendation System
        pivot_table = recommendation.prepare_data(df_recommendation)  # create pivot table
        model = recommendation.train_model(pivot_table)  # train model
        recommendation.save_similar_products(input_product_id, pivot_table, model, df_recommendation, similar_products_path)  # save scores


if __name__ == "__main__":
    run_pipeline()
