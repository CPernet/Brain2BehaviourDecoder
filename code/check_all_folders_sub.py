import os
import shutil
import pandas as pd

# Define paths
participants_file = "participants.tsv"
problematic_folder = "problematic"

# Create 'problematic' folder if it doesn't exist
os.makedirs(problematic_folder, exist_ok=True)

# Load the participants.tsv file
df = pd.read_csv(participants_file, sep="\t")

# Check for required 'ID' column
if "ID" not in df.columns:
    raise ValueError("The 'ID' column is missing from participants.tsv")

# Iterate through each row
for _, row in df.iterrows():
    participant_id = "sub-"+str(row["ID"])
    participant_folder = os.path.join(os.getcwd(), participant_id)

    # Check if any value in the row is NaN
    has_nan = row.isna().any()

    # Check if the folder exists
    if not os.path.exists(participant_folder) or has_nan:
        destination = os.path.join(problematic_folder, participant_id)

        # Move the folder if it exists
        if os.path.exists(participant_folder):
            shutil.move(participant_folder, destination)
            print(f"Moved {participant_folder} to {destination}")
        else:
            print(f"Folder missing for ID {participant_id}")

print("Processing complete.")
