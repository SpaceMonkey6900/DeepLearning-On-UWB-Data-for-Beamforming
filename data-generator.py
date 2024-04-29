import numpy as np
import pandas as pd

# Room dimensions
room_length = 10
room_breadth = 10
room_height = 5

# Smartphone coordinates
smartphone_x = 5
smartphone_y = 8
smartphone_z = 2

# Anchor coordinates
anchor1_coords = np.array([0, 0, 0])
anchor2_coords = np.array([room_length, 0, 0])
anchor3_coords = np.array([0, room_breadth, 0])

# Number of data points to generate
num_data_points = 1000

# Generate synthetic UWB data
data = []
for _ in range(num_data_points):
    # Add random noise to smartphone coordinates
    noise_x = np.random.normal(0, 0.1)
    noise_y = np.random.normal(0, 0.1)
    noise_z = np.random.normal(0, 0.1)
    
    smartphone_coords = np.array([smartphone_x + noise_x, smartphone_y + noise_y, smartphone_z + noise_z])
    
    # Calculate distances from smartphone to anchors
    dist1 = np.linalg.norm(smartphone_coords - anchor1_coords)
    dist2 = np.linalg.norm(smartphone_coords - anchor2_coords)
    dist3 = np.linalg.norm(smartphone_coords - anchor3_coords)
    
    # Add random noise to distance measurements
    noise_dist1 = np.random.normal(0, 0.05)
    noise_dist2 = np.random.normal(0, 0.05)
    noise_dist3 = np.random.normal(0, 0.05)
    
    dist1 += noise_dist1
    dist2 += noise_dist2
    dist3 += noise_dist3
    
    # Append data to the list
    data.append([dist1, dist2, dist3, smartphone_x + noise_x, smartphone_y + noise_y, smartphone_z + noise_z])

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['anchor1_dist', 'anchor2_dist', 'anchor3_dist', 'smartphone_x', 'smartphone_y', 'smartphone_z'])

# Save the data to a CSV file
df.to_csv('synthetic_uwb_data.csv', index=False)