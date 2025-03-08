import os

def ensure_folders_exist():
    """Ensure all required folders exist."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data")
    
    # Create data folder if it doesn't exist
    if not os.path.exists(data_path):
        print(f"Creating data folder at {data_path}")
        os.makedirs(data_path)
        print("✅ Data folder created successfully")
        print("👉 Add your PDF files to this folder")
    else:
        print(f"✅ Data folder already exists at {data_path}")

if __name__ == "__main__":
    ensure_folders_exist()