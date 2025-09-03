import pandas as pd
import os

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with basic validation.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    # Convert to absolute path
    path = os.path.abspath(path)
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File does not exist: {path}")
    
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Loaded {os.path.basename(path)} with shape {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to read CSV: {e}")


# =========================
# Example usage (optional)
# =========================
if __name__ == "__main__":
    # Adjust relative path to your raw dataset
    raw_path = os.path.join("data", "raw", "stroke_data.csv")
    df = load_csv(raw_path)
    print(df.head())
