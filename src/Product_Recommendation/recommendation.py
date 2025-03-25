import numpy as np
from sklearn.neighbors import NearestNeighbors


def prepare_data(df):
    df['review/score'] = df['review/score'].astype(float)
    pivot_table = df.pivot_table(index='product/productId', columns='review/userId', values='review/score')  # create pivot table
    pivot_table = pivot_table.fillna(0)  # fill missing values with 0
    return pivot_table


def train_model(pivot_table):
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)  # load model with metric cosine
    knn.fit(pivot_table.values)  # train model
    return knn  # return trained model


def save_similar_products(product_id,pivot_table,knn,df,file_path, n=5):
    product_index = pivot_table.index.get_loc(product_id)  # get the index of the product from pivot table
    distances, indices = knn.kneighbors([pivot_table.iloc[product_index].values], n_neighbors=n + 1)  # find similar products
    similar_products = pivot_table.index[indices.flatten()[1:]]
    similar_product_titles = df.loc[df['product/productId'].isin(similar_products), 'product/title'].values  # get the titles of the dataframe
    similar_product_titles = np.unique(similar_product_titles)  # get only unique titles
    with open(file_path, "w") as file:
        for title in similar_product_titles:
            file.write(title + "\n")
