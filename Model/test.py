import pandas as pd

# File paths
input_csv = "dataset.csv"  # Replace with your CSV file path
train_csv = "train.csv"
test_csv = "test.csv"

# Parameters
letters = list("ABCDEFGHIKLMNOPQRSTUVWXY#")


# Read the CSV
print("Reading CSV...")
df = pd.read_csv(input_csv)

# Validate structure
expected_cols = ["letter", "hand"] + [f"{coord}{i}" for i in range(21) for coord in ["x", "y", "z"]]
if list(df.columns) != expected_cols:
	raise ValueError(f"Expected columns {expected_cols}, got {list(df.columns)}")

# Split into train and test
print("Splitting train/test...")
train_data = []
test_data = []

for letter in letters:
	letter_data = df[df["letter"] == letter]

	train_sample = letter_data.sample(n=int(len(letter_data) * 0.8), random_state=42)
	remaining = letter_data.drop(train_sample.index)
	test_sample = remaining.sample(n=len(remaining), random_state=42)

	train_data.append(train_sample)
	test_data.append(test_sample)

# Combine and shuffle
train_df = pd.concat(train_data, ignore_index=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
test_df = pd.concat(test_data, ignore_index=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Save to separate CSVs
print("Saving to train.csv and test.csv...")
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"Done! Train rows: {len(train_df)}, Test rows: {len(test_df)}")
