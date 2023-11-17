import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('C:/Users/kma/Documents/NYU/2023-24 Senior Year/Predictive Analytics/HW3/NBA_Player_Stats.csv')

# Splitting the dataset into features and targets
X = df.iloc[:, :-5]  # All columns except the last five
y = df.iloc[:, -5:]  # Only the last five columns

def euclidean_distance(row1, row2):
    """
    Calculate the Euclidean distance between two rows.
    """
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
    """
    Locate the most similar neighbors.
    """
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def rmse(predicted, actual):
    return np.sqrt(np.mean((predicted - actual) ** 2))

def predict_multioutput_regression(train, test_row, num_neighbors):
    """
    Make a prediction with neighbors for multiple output regression.
    """
    neighbors = get_neighbors(train, test_row, num_neighbors)
    
    # Split neighbors into features and targets
    neighbor_features = [row[:-5] for row in neighbors]
    neighbor_targets = [row[-5:] for row in neighbors]

    predictions = []
    for i in range(5):  # Loop over each target variable
        output_values = [target[i] for target in neighbor_targets]
        prediction = sum(output_values) / float(len(output_values))
        predictions.append(prediction)

    return predictions

# Convert the DataFrame to a list of lists
dataset = pd.concat([X, y], axis=1).values.tolist()

# Split dataset (you can use a more sophisticated method for splitting)
train_set = dataset[:int(len(dataset) * 0.8)]
test_set = dataset[int(len(dataset) * 0.8):]

# Set the number of neighbors
k = 3

# Making predictions for each row in the test set
predictions = []
for row in test_set:
    output = predict_multioutput_regression(train_set, row, k)
    predictions.append(output)

#print(np.shape(predictions)) #Correspond to 426 active players as row
#for row in predictions:      #Print predictions
#    print(row)



