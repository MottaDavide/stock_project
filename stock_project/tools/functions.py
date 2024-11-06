import pandas as pd
import re
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Set, List



def find_first_matching_transaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the first matching non-cancelled transaction for each cancelled transaction in the DataFrame.
    
    This function searches for matching transactions where:
      - `stock_code` and `customer_id` are the same.
      - The `qty` in the non-cancelled transaction is the opposite (negated) of the cancelled transaction's `qty`.
    
    The function returns the original DataFrame with rows removed that represent pairs of matched transactions.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing transaction data. 
                           Must include columns: 'stock_code', 'customer_id', 'qty', 'is_cancelled', and 'invoice_id'.
    
    Returns:
        pd.DataFrame: The DataFrame with matched (cancelled and non-cancelled) transaction rows removed.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()
    
    # Reset the index to have a unique identifier for each row
    df_temp = df.copy()
    df_temp = df_temp.reset_index()

    # Create a unique transaction ID by concatenating stock_code, customer_id, and qty
    df_temp['transaction_id'] = df_temp['stock_code'] + '_' + df_temp['customer_id'].astype(str) + '_' + df_temp['qty'].astype(str)
    
    # Separate cancelled and non-cancelled transactions
    cancelled = df_temp[df_temp['is_cancelled'] == True]
    non_cancelled = df_temp[df_temp['is_cancelled'] == False]
    
    # Create a 'match_id' in the cancelled transactions where the qty is negated (for matching opposite quantities)
    cancelled['match_id'] = cancelled['stock_code'] + '_' + cancelled['customer_id'].astype(str) + '_' + (-cancelled['qty']).astype(str)
    
    # Merge the cancelled transactions with non-cancelled ones based on matching stock_code, customer_id, and opposite qty
    matches = pd.merge(
        cancelled[['invoice_id', 'stock_code', 'customer_id', 'qty', 'match_id', 'index']],
        non_cancelled[['invoice_id', 'transaction_id', 'index']],
        left_on='match_id',
        right_on='transaction_id',
        how='inner'
    )
    
    # Rename the columns of the resulting DataFrame to make the matching process clearer
    matches.columns = ['cancelled_invoice', 'stock_code', 'customer_id', 'qty', 'match_id', 'cancelled_index', 'matching_invoice', 'transaction_id', 'match_index']


    # Group by the cancelled index to ensure we get the first matching transaction for each cancelled transaction
    matches = matches.groupby('cancelled_index', as_index=False).first()

    
    # Combine the indices of both the cancelled and matching transactions
    lst_index = matches['cancelled_index'].tolist() + matches['match_index'].tolist()
    
    
    
    n = df.shape[0]
    df = df[~df.index.isin(lst_index)]
    
    #print(f"Number of matching: {matches.shape[0]}")
    #print(f"Number of dropped records: {n - df.shape[0]}")
    return df

from typing import Set

def find_odd_code(df: pd.DataFrame) -> List[str]:
    """
    Identifies and returns a set of stock codes in the DataFrame that match certain irregular patterns
    based on a regular expression search.

    The patterns used to identify "odd" codes include:
    - A single letter (e.g., 'A', 'B').
    - Non-alphanumeric characters before or after uppercase letters.
    - Mixed letter and digit combinations (e.g., 'A1', 'B2').
    - Non-alphanumeric characters followed by lowercase letters.

    Args:
        df (pd.DataFrame): The DataFrame containing a 'stock_code' column with the stock codes to analyze.

    Returns:
        Set[str]: A set of unique stock codes that match the irregular patterns.
    
    Example:
        >>> odd_codes = find_odd_code(df)
        >>> print(odd_codes)
        {'A', 'B2', '-C', 'x1'}
    """
    
    # Initialize a set to store unique matches (sets avoid duplicates)
    spec_list = set()

    # Iterate through each stock code in the DataFrame
    for code in df['stock_code']:
        # Use regular expression to find patterns
        matches = re.findall(r"^\w{1}$|\D[A-Z]+\D|[A-Z]\d|\D[a-z]", code)
        
        # If matches are found, add them to the set
        if matches:
            spec_list.update(matches)  # Add multiple matches at once if found
    
    return list(spec_list)

def optimal_kmeans(X: np.ndarray, max_clusters: int = 10) -> int:
    """
    Determines the optimal number of clusters for KMeans clustering using the silhouette score as the evaluation metric.
    The function fits KMeans models with different numbers of clusters, calculates the silhouette score for each model, 
    and returns the optimal number of clusters.

    Args:
        X (np.ndarray): The input data to be clustered (n_samples, n_features).
        max_clusters (int, optional): The maximum number of clusters to test. Defaults to 10.

    Returns:
        int: The optimal number of clusters based on the highest silhouette score.
    
    Raises:
        ValueError: If `max_clusters` is less than 2, or if the input array `X` has insufficient samples.

    Example:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=300, centers=5, random_state=42)
        >>> optimal_k = optimal_kmeans(X, max_clusters=10)
        >>> print(f"Optimal number of clusters: {optimal_k}")
    """
    
    if max_clusters < 2:
        raise ValueError("max_clusters must be at least 2.")
    if len(X) < 2:
        raise ValueError("Input data must have at least 2 samples.")
    
    silhouette_scores = []  # List to store silhouette scores
    cluster_range = range(2, max_clusters + 1)  # Range of cluster values to try

    # Iterate through possible number of clusters and compute silhouette scores
    for n_clusters in cluster_range:
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate the average silhouette score for the current number of clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        # Optionally, you could compute individual sample silhouette values for further analysis
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        print(f"For n_clusters = {n_clusters}, the average silhouette score is {silhouette_avg:.4f}")
        
    # Plot the silhouette scores for each number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title('Silhouette Scores for different K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    # Find the number of clusters that resulted in the highest silhouette score
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters: {optimal_clusters}")
    
    return optimal_clusters

def item_cluster(df_model: pd.DataFrame) -> pd.DataFrame:
    """
    Groups products into clusters based on their price and quantity statistics (mode, median, last price, average quantity),
    using KMeans clustering. The function standardizes the data, applies a Box-Cox transformation, and determines the optimal
    number of clusters using silhouette scores.

    Args:
        df_model (pd.DataFrame): A DataFrame containing product data with 'stock_code', 'price_mode', 'price_median', 
            'price_last', and 'qty' columns.

    Returns:
        pd.DataFrame: The modified `df_model` DataFrame with an additional 'product_type' column that stores the 
        cluster assignment for each stock code.
    
    Example:
        >>> df_clustered = item_cluster(df_model)
        >>> print(df_clustered.head())
    """
    
    # Select the relevant product information and drop duplicates
    df_product = df_model[['stock_code', 'price_mode', 'price_median', 'price_last']].drop_duplicates().set_index('stock_code')

    # Calculate average and median quantity per product (stock_code)
    df_product = df_product.merge(
        df_model.groupby(by='stock_code').agg(qty_avg=('qty', 'mean'), qty_median=('qty', 'median')), 
        left_index=True, right_index=True
    )

    # Make a copy of the original product DataFrame for later use
    df_product_original = df_product.copy()

    # Apply Box-Cox transformation to make the data more Gaussian-like
    for col in df_product.columns:
        df_product[col] = boxcox(df_product[col])[0]
    
    # Standardize the transformed data
    X = df_product.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine the optimal number of clusters using silhouette scores
    k = optimal_kmeans(X_scaled, max_clusters=10)
    
    # Perform KMeans clustering
    km = KMeans(n_clusters=k, random_state=42)
    df_product["cluster"] = km.fit_predict(X_scaled)
    df_product_original["cluster"] = df_product["cluster"]  # Preserve cluster assignments

    # Map product cluster to the original df_model
    df_model["product_type"] = df_model.stock_code.map(df_product["cluster"])
    df_model["product_type"] = df_model["product_type"].astype("object")
    
    return df_model

def series_cluster(df_model: pd.DataFrame, max_cluster: int = 3) -> pd.DataFrame:
    """
    Groups time series data for each stock code into clusters based on the sales quantity over time,
    using hierarchical clustering. The function normalizes the time series data, calculates pairwise distances
    using Ward's method, and assigns stock codes to clusters.

    Args:
        df_model (pd.DataFrame): A DataFrame containing stock code sales data with 'date', 'stock_code', and 'qty' columns.
        max_cluster (int, optional): The maximum number of clusters for hierarchical clustering. Defaults to 3.

    Returns:
        pd.DataFrame: The modified `df_model` DataFrame with an additional 'time_series_type' column that stores the 
        cluster assignment for each stock code.
    
    Example:
        >>> df_clustered = series_cluster(df_model, max_cluster=3)
        >>> print(df_clustered.head())
    """
    
    # Pivot the DataFrame to create time series data for each stock_code
    pivot_df = df_model.pivot_table(index='date', columns='stock_code', values='qty', aggfunc='sum', fill_value=0)

    # Normalize the time series data (standardize each time series)
    pivot_df_normalized = (pivot_df - pivot_df.mean()) / pivot_df.std()

    # Perform hierarchical clustering using Ward's method
    Z = linkage(pivot_df_normalized.T, method='ward')  # Transpose to cluster based on stock_code

    # Plot dendrogram to visualize the hierarchical clustering
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram for Time Series Clustering")
    dendrogram(Z, labels=pivot_df.columns)
    plt.show()

    # Assign clusters based on the desired number of clusters (max_cluster)
    clusters = fcluster(Z, t=max_cluster, criterion='maxclust')

    # Create a DataFrame to store the cluster assignments for each stock_code
    df_cluster = pd.DataFrame({
        'stock_code': pivot_df.columns,
        'cluster': clusters
    })

    # Map time series cluster assignments back to the original df_model
    df_model["time_series_type"] = df_model.stock_code.map(df_cluster.set_index('stock_code')['cluster'])
    df_model["time_series_type"] = df_model["time_series_type"].astype("object")
    
    return df_model