# Load and preprocess the data
data = pd.read_csv('News.csv', index_col=0)
data = data.drop(["subject", "date"], axis=1)
data = data.sample(frac=1).reset_index(drop=True)

# Ensure class is of type int
data['class'] = data['class'].astype(int)

