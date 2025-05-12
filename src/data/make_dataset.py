import pandas as pd


# Use raw string or forward slashes to avoid path issues
file_path = r'data\processed\data.xlsx'  # OR 'data/processed/data.xlsx'

# Load Excel file (make sure openpyxl is installed)
df = pd.read_excel(file_path)

# Check the columns (debugging step)
print("Columns in the dataset:", df.columns)

# Define a function to format the data for GPT-2 training
def format_conversation(row):
    return f"<|startoftext|> User: {row['Input']} <|endoftext|> Response: {row['Response']} <|endoftext|>"

# Apply the formatting function to each row
formatted_data = df.apply(format_conversation, axis=1)

# Save the formatted data to a text file for training
with open('formatted_conversations.txt', 'w', encoding='utf-8') as f:
    for conversation in formatted_data:
        f.write(conversation + '\n')

print("Formatted data saved to 'formatted_conversations.txt'")