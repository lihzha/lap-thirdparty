import pickle
import numpy as np

# Replace with your actual pickle file path
pickle_file_path = 'data.pkl'

# Load the pickle file
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

breakpoint()

# Check if the data is a dictionary-like object
if isinstance(data, dict):
    print(f"Loaded pickle file with {len(data)} keys.\n")
    for key, value in data.items():
        # Try to print shape if available
        try:
            shape = value.shape
        except AttributeError:
            shape = 'No shape attribute'
        print(f"Key: {key} | Type: {type(value)} | Shape: {shape}")
else:
    print("The loaded object is not a dictionary.")
    print(f"Type: {type(data)}")
    # If it's a list of items, you can optionally inspect each one:
    if isinstance(data, list):
        for i, item in enumerate(data):
            try:
                shape = item.shape
            except AttributeError:
                shape = 'No shape attribute'
            print(f"Index: {i} | Type: {type(item)} | Shape: {shape}")
