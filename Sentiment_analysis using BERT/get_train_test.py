import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('data.csv')

# Separate the data into two groups: label=0 and label=1
label_0 = data[data['label'] == 0]
label_1 = data[data['label'] == 1]

# Split the data into train and test sets for each label group, using an 80:20 ratio
train_label_0, test_label_0 = train_test_split(label_0, test_size=0.2, random_state=42)
train_label_1, test_label_1 = train_test_split(label_1, test_size=0.2, random_state=42)

# Concatenate the train and test sets for each label group into final train and test sets
train_data = pd.concat([train_label_0, train_label_1])
test_data = pd.concat([test_label_0, test_label_1])

# Shuffle the final train and test sets to ensure randomness
train_data = train_data.sample(frac=1, random_state=42)
test_data = test_data.sample(frac=1, random_state=42)

# Save the train and test sets as CSV files
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)
