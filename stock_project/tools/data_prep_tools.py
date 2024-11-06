
import pandas as pd
from pathlib import Path
import numpy as np

from stock_project.tools.global_variables import PATH_DATA_INPUT
from stock_project.tools.functions import find_first_matching_transaction, find_odd_code, item_cluster, series_cluster


def original_to_csv(
    path_to_original: str | Path,
    path_to_save: str | Path | None = None,
    **kwargs
) -> pd.DataFrame:
    """
    Reads an Excel file with multiple sheets, concatenates the data, and optionally saves it as a CSV file.

    Args:
        path_to_original (str | Path): Path to the original Excel file.
        path_to_save (str | Path | None, optional): Path where the concatenated CSV file will be saved.
            If None, the CSV is not saved. Defaults to None.
        **kwargs: Additional arguments to be passed to `pd.DataFrame.to_csv()` (e.g., `sep`, `index`, etc.).

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data from all sheets of the Excel file.

    Raises:
        FileNotFoundError: If the provided Excel file path does not exist.

    Example:
        # To load data from an Excel file and save as CSV:
        >>> df = original_to_csv("data/my_excel_file.xlsx", "data/my_data.csv", index=False)
        
        # To load data without saving to CSV:
        >>> df = original_to_csv("data/my_excel_file.xlsx")
    """
    
    # Load the Excel file containing multiple sheets
    df_original = pd.ExcelFile(path_to_original)
    
    # Read each sheet and store them in a list of DataFrames
    dfs = [pd.read_excel(df_original, sheet_name=sheet, dtype=object) for sheet in df_original.sheet_names]
    
    # Concatenate all the DataFrames from the sheets into one DataFrame
    df_raw = pd.concat(dfs, ignore_index=True)
    
    # If a save path is provided, save the concatenated DataFrame as a CSV
    if path_to_save:
        df_raw.to_csv(path_to_save, **kwargs)
    
    # Return the concatenated DataFrame
    return df_raw


def data_cleaning(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Cleans a DataFrame or loads a dataset from a CSV file, applying several transformations 
    such as renaming columns, cleaning stock codes, and filtering out invalid records.
    The cleaned DataFrame is saved as a CSV file and returned.

    Args:
        df (pd.DataFrame | None, optional): A DataFrame to be cleaned. 
            If None, the function will load the dataset from a CSV file located at `PATH_DATA_INPUT / 'raw/data.csv'`.
            Defaults to None.

    Returns:
        pd.DataFrame: The cleaned DataFrame with all transformations applied.

    Raises:
        FileNotFoundError: If the CSV file is not found when `df` is None.

    Notes:
        - The function assumes that the dataset contains columns such as 'Invoice', 'StockCode', 'Description',
          'Quantity', 'Price', 'InvoiceDate', 'Customer ID', and 'Country'.
        - A CSV of the cleaned dataset is saved to `PATH_DATA_INPUT / 'clean/data_clean.csv'`.
        - Any row with invalid or erroneous data is dropped, including those with negative quantities, prices, 
          or descriptions with less than seven characters.

    Cleaning Process:
        1. Column Renaming: Rename columns to more meaningful names.
        2. Date Processing: Convert invoice date into datetime and create separate date fields.
        3. Description Normalization: Convert descriptions to lowercase and strip whitespaces.
        4. Remove duplicates and invalid transactions.
        5. Clean stock codes and descriptions with multiple values.
        6. Filter out invalid or extreme quantities and prices.
        7. Remove cancelled invoices and transactions with abnormal stock codes.

    Example:
        # Cleaning a DataFrame already loaded:
        >>> df_cleaned = data_cleaning(df)
        
        # Cleaning a dataset from CSV file:
        >>> df_cleaned = data_cleaning()

    """
    
    # If no DataFrame is provided, load it from the default path
    if not df:
        schema = {
            'Invoice': str,
            'StockCode': str,
            'Description': str,
            'Quantity': np.int32,
            'Price': np.float32,
            'InvoiceDate': str,
            'Customer ID': str,
            'Country': str
        }
        df = pd.read_csv(PATH_DATA_INPUT / 'raw/data.csv', sep='\t', dtype=schema, parse_dates=True, encoding='ISO-8859-1')

    df = df.copy()  # Avoid modifying the original dataframe
    
    print("Start Cleaning Process...")
    n = df.shape[0]  # Store initial number of rows for comparison later
    
    # Step 1: Renaming column names for better clarity
    df = df.rename(columns={
        'Invoice': 'invoice_id',
        'StockCode': 'stock_code',
        'Description': 'description',
        'Quantity': 'qty',
        'InvoiceDate': 'invoice_timestamp',
        'Price': 'unit_price',
        'Customer ID': 'customer_id',
        'Country': 'country'
    })
    
    # Step 2: Convert the 'invoice_timestamp' to a datetime object and create a new 'invoice_date' column
    df['invoice_timestamp'] = pd.to_datetime(df['invoice_timestamp'], format='%Y-%m-%d %H:%M:%S')
    df['invoice_date'] = pd.to_datetime(df['invoice_timestamp'].dt.date)
    
    # Step 3: Clean description by lowering and stripping white spaces
    df['description'] = df['description'].str.lower().str.strip()
    
    # Step 4: Drop duplicates in the data
    df_clean = df.drop_duplicates()
    
    # CLEANING INVOICE
    print("CLEANING INVOICE")
    # Step 5: Mark cancelled invoices (those that start with 'C')
    df_clean["is_cancelled"] = np.where(df_clean.invoice_id.str.startswith("C"), True, False)
    
    # Identify and drop invalid transactions
    df_temp = df_clean.assign(len_invoice=df_clean['invoice_id'].str.len())
    valid_odd_trans = df_temp[(df_temp.is_cancelled == False) & (df_temp.len_invoice == 7)]['invoice_id']
    df_clean = df_clean[~df_clean.invoice_id.isin(valid_odd_trans)]
    
    # Clean based on first matching transactions (custom logic)
    df_clean = find_first_matching_transaction(df_clean)
    
    # Remove cancelled invoices
    df_clean = df_clean[df_clean['is_cancelled'] == False].drop(columns='is_cancelled')
    
    # CLEANING STOCK CODE
    print("CLEANING STOCK CODE")
    # Remove invalid stock codes based on custom logic
    df_clean = df_clean[~df_clean['stock_code'].str.startswith(tuple(find_odd_code(df_clean)))]
    df_clean['stock_code'] = df_clean['stock_code'].str.upper()
    
    # CLEANING DESCRIPTION
    print("CLEANING DESCRIPTION")
    # Identify stock codes with multiple descriptions and standardize them
    df_temp = df_clean.groupby(by='stock_code', as_index=False)['description'].nunique()
    lst_multiple_desc = df_temp[df_temp['description'] > 1]['stock_code'].unique().tolist()
    
    dict_replace_description = (df_clean[df_clean['stock_code'].isin(lst_multiple_desc)]
                                .groupby(by='stock_code', as_index=False)['description']
                                .agg(lambda x: pd.Series.mode(x)[0])
                                .set_index('stock_code')['description']
                                .to_dict())
    
    # Replace inconsistent descriptions
    df_clean = df_clean.assign(description=df_clean.stock_code.map(dict_replace_description).fillna(df_clean.description))
    df_clean = df_clean[~df_clean.description.isna()]
    df_clean = df_clean[~(df_clean['description'].str.len() <= 7)]
    
    # CLEANING QUANTITY
    print("CLEANING QUANTITY")
    # Remove invalid quantities (negative or extreme values)
    df_clean = df_clean[df_clean['qty'] > 0]
    df_clean = df_clean[~(df_clean.description.str.startswith(('set of', 'pack of')))]
    df_clean = df_clean[df_clean.qty < df_clean.qty.quantile(0.995)]
    
    # CLEANING PRICE
    print("CLEANING PRICE")
    # Remove invalid prices (negative or extreme values)
    df_clean = df_clean[df_clean['unit_price'] > 0]
    df_clean = df_clean[df_clean.unit_price < df_clean.unit_price.quantile(0.9999)]
    
    # Renaming columns and removing unused columns
    df_clean = df_clean.rename(columns={
        'unit_price': 'price',
        'invoice_date': 'date'
    })
    df_clean = df_clean.drop(columns='invoice_timestamp')
    
    # Save the cleaned DataFrame to CSV
    df_clean.to_csv(PATH_DATA_INPUT / "clean/data_clean.csv", sep="\t", index=False)
    
    # Output how many records were dropped
    print(f"We dropped {n - df_clean.shape[0]} records\n")
    print("End!")
    
    return df_clean



def data_model_preparation(df: pd.DataFrame | None = None, level: str = 'day') -> pd.DataFrame:
    """
    Prepares a cleaned DataFrame for model training by performing feature engineering 
    such as lag creation, date manipulation, price aggregation, and aggregating sales quantities
    at different levels (day or week).

    Args:
        df (Optional[pd.DataFrame], optional): A cleaned DataFrame to be processed.
            If None, the function will call `data_cleaning()` to load and clean the dataset.
            Defaults to None.
        level (str, optional): The aggregation level of the data. 
            Can be 'day' (default) or 'week'. If 'week', data will be aggregated by week.

    Returns:
        pd.DataFrame: A DataFrame with engineered features, including lagged quantities, 
        price statistics, date features, and grouping by the selected aggregation level.
    
    Raises:
        ValueError: If the `level` argument is not one of 'day' or 'week'.

    Example:
        # Prepare data at daily level:
        >>> df_prepared = data_model_preparation(df, level='day')
        
        # Prepare data at weekly level:
        >>> df_prepared = data_model_preparation(df, level='week')
    """
    
    if df is None:
        df = data_cleaning()  # Load and clean the dataset if no DataFrame is provided
    
    df_model = df.copy()  # Work with a copy to avoid modifying the original DataFrame
    
    print("\nPreparing data for the model...")
    
    # Reordering columns to put 'qty' as the first column
    cols = ['qty'] + [col for col in df_model.columns if col != 'qty']
    df_model = df_model[cols]
    
    # FEATURE ENGINEERING
    print(" FEATURE ENGINEERING")
    
    # Date Features
    df_model['day'] = df_model['date'].dt.day
    df_model['day_of_week'] = df_model['date'].dt.dayofweek
    df_model['day_of_year'] = df_model['date'].dt.dayofyear
    df_model['week'] = df_model['date'].dt.isocalendar().week
    df_model['month'] = df_model['date'].dt.month
    df_model['year'] = df_model['date'].dt.year
    
    # Price Features: Calculate mode, median, and last price for each stock_code
    df_model['price_mode'] = df_model.groupby('stock_code')['price'].transform(lambda x: pd.Series.mode(x)[0])
    df_model['price_median'] = df_model.groupby('stock_code')['price'].transform('median')
    df_model['price_last'] = df_model.groupby('stock_code')['price'].transform('last')
    
    # Description Length Feature
    df_model['description_len'] = df_model['description'].str.len()
    
    # Lag Features: Number of invoices (Lag of 1 month)
    df_model = df_model.sort_values(by='date')
    
    # Lag on monthly invoices
    df_temp = df_model.groupby(by=['year', 'month'], as_index=False).agg(n_invoice=('invoice_id', 'nunique'))
    df_temp['n_invoice_lag1m'] = df_temp['n_invoice'].shift(1)
    df_model = df_model.merge(df_temp[['year', 'month', 'n_invoice_lag1m']], on=['year', 'month'], how='left')
    
    # Lag on monthly invoices at the stock_code level
    df_temp = df_model.groupby(by=['stock_code', 'year', 'month'], as_index=False).agg(n_invoice=('invoice_id', 'nunique'))
    df_temp['n_invoice_lag1m_stock'] = df_temp.groupby('stock_code')['n_invoice'].shift(1)
    df_model = df_model.merge(df_temp[['stock_code', 'year', 'month', 'n_invoice_lag1m_stock']], on=['stock_code', 'year', 'month'], how='left')

    # Dropping columns related to customer, country, and invoice_id based on the aggregation level
    if level == 'day':
        df_model = df_model.groupby(by=[col for col in df_model.columns if col not in ['customer_id', 'country', 'invoice_id', 'qty']], 
                                    as_index=False)['qty'].sum()
    elif level == 'week':
        # Convert date to start of the week (Monday)
        df_model['date'] = df_model['date'].dt.to_period('W').apply(lambda r: r.start_time)
        # Drop day-level columns
        df_model = df_model.drop(columns=['day', 'day_of_year', 'day_of_week'])
        df_model = df_model.groupby(by=[col for col in df_model.columns if col not in ['customer_id', 'country', 'invoice_id', 'qty']], 
                                    as_index=False)['qty'].sum()
    else:
        raise ValueError("Invalid level argument: choose either 'day' or 'week'")
    
    # Peak Season Feature (Sept-Nov as peak season)
    df_model['peak_season'] = df_model['month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)
    
    # Product Type Feature using a custom item clustering function
    df_model = item_cluster(df_model)
    
    # Time Series Type Feature using a custom series clustering function
    df_model = series_cluster(df_model)
    
    # Lag Features for Quantity at the stock_code level
    df_model['qty_lag1d'] = df_model.groupby('stock_code')['qty'].shift(1)   # Lag by 1 day
    df_model['qty_lag5d'] = df_model.groupby('stock_code')['qty'].shift(5)   # Lag by 5 days
    df_model['qty_lag7d'] = df_model.groupby('stock_code')['qty'].shift(7)   # Lag by 7 days
    df_model['qty_lag30d'] = df_model.groupby('stock_code')['qty'].shift(30) # Lag by 30 days
    
    return df_model
    
    
    