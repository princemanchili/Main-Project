# import pandas as pd
# import numpy as np

# # Define the categories for Pneumonia
# pneumonia_categories = ['No Pneumonia', 'Mild Pneumonia', 'Moderate Pneumonia', 'Severe Pneumonia']

# # Generate synthetic data
# num_entries = 300

# data = {
#     'Age': np.random.randint(1, 300, size=num_entries),
#     'Gender': np.random.choice(['Male', 'Female'], size=num_entries),
#     'Body_Temperature': np.random.uniform(98.0, 103.0, size=num_entries),
#     'Cough': np.random.choice([0, 1], size=num_entries),
#     'Sore_Throat': np.random.choice([0, 1], size=num_entries),
#     'Difficulty_Breathing': np.random.choice([0, 1], size=num_entries),
#     'Chest_Pain': np.random.choice([0, 1], size=num_entries),
#     'White_Blood_Cell_Count': np.random.randint(4000, 12001, size=num_entries),
#     'Heart_Rate': np.random.randint(60, 101, size=num_entries),
#     'Respiratory_Rate': np.random.randint(12, 21, size=num_entries),
#     'Pneumonia': np.random.choice(pneumonia_categories, size=num_entries)
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Display the first few rows of the DataFrame
# print(df.head())

# # Save the DataFrame to a CSV file
# df.to_csv('pneumonia_dataset1.csv', index=False)



import pandas as pd
import numpy as np

# Define the categories for Pneumonia
pneumonia_categories = ['No Pneumonia', 'Mild Pneumonia', 'Moderate Pneumonia', 'Severe Pneumonia']

# Generate synthetic data
num_entries = 8000

data = {
    'Age': np.random.randint(1, 300, size=num_entries),
    'Gender': np.random.choice(['Male', 'Female'], size=num_entries),
    'Body_Temperature': np.random.uniform(98.0, 103.0, size=num_entries),
    'Cough': np.random.choice([0, 1], size=num_entries),
    'Sore_Throat': np.random.choice([0, 1], size=num_entries),
    'Difficulty_Breathing': np.random.choice([0, 1], size=num_entries),
    'Chest_Pain': np.random.choice([0, 1], size=num_entries),
    'White_Blood_Cell_Count': np.random.randint(4000, 12001, size=num_entries),
    'Heart_Rate': np.random.randint(60, 101, size=num_entries),
    'Respiratory_Rate': np.random.randint(12, 21, size=num_entries),
    'Pneumonia': np.random.choice(pneumonia_categories, size=num_entries)
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv('pneumonia_dataset3.csv', index=False)