import pandas as pd
from pathlib import Path

def preprocess_data(input_filepath: Path, output_filepath: Path):
    """
    Load raw stock data CSV, clean, and save processed CSV.
    
    Args:
        input_filepath (Path): Path to raw CSV file.
        output_filepath (Path): Path to save cleaned CSV.
    """
    df = pd.read_csv(input_filepath)
    df.dropna(inplace=True)  # Remove missing values
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df.to_csv(output_filepath)
    print(f"Saved cleaned data to {output_filepath}")

if __name__ == "__main__":
    # Resolve paths relative to this script file location
    base_dir = Path(__file__).parent.parent  # project root
    raw_file = base_dir / 'data' / 'raw' / 'AAPL_stock_data.csv'
    processed_file = base_dir / 'data' / 'processed' / 'AAPL_stock_data_clean.csv'

    preprocess_data(raw_file, processed_file)
