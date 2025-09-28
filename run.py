import pandas as pd

DATA_PATH = r"C:\Users\3 Stars Laptop\OneDrive\Desktop\predictionwebsite\data\Combined_Gold_Silver_Data.csv"
df = pd.read_csv(DATA_PATH)
print(df.columns.tolist())
