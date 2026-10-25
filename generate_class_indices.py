import os
import json

train_dir = 'data/train'  # Make sure this points to your correct training folder

# Get class names from folders
class_names = sorted(os.listdir(train_dir))
class_indices = {name: idx for idx, name in enumerate(class_names)}

# Save the indices to a JSON file
os.makedirs('models', exist_ok=True)
with open('models/class_indices.json', 'w') as f:
    json.dump(class_indices, f)

print("✅ class_indices.json has been created in the 'models' folder.")
print("📦 Classes found:", class_indices)
