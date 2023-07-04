import os
from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM


def create_sequential_dataset(directory):

    dataset = []  # List to store the loaded dataframes
    labels = []
    # Iterate over files in the directory
    for filename in tqdm(os.listdir(directory)):
        try:
            if filename.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(directory, filename)
                f = open(file_path, "r")
                label = ''
                data = []
                for line in f.readlines():
                    items = line.split(",")
                    if items[0] == '"ACTIVITY TYPE:"':
                        label = [1 if items[1].strip() == '"Running"' else 0]
                    if len(items) == 4 and items[0]!='"Time [sec]"':
                        data.append([float(item.replace("\"","").strip()) for item in items])
                dataset.append(data[:91])
                labels.append(label)
        except:
            pass



    return dataset, labels


new_dataset, labels = create_sequential_dataset(r"dataIOT")




# Define the training data
# Assuming each data sample has 10 timesteps and 3 features
data_samples = np.array(new_dataset)
data_labels = np.array(labels)

# Define the corresponding labels (0 for walking, 1 for running)
labels = np.array(labels)

# Define the RNN model
model = Sequential()
model.add(LSTM(10, input_shape=(91,4)))  # 64 is the number of units in the LSTM layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(data_samples, data_labels, epochs=30, batch_size=1)

model.save('walk_run_model.mdl')
predictions = model.predict(np.array([data_samples[0]]))
predicted_labels = ['walking' if pred < 0.5 else 'running' for pred in predictions]

# Print the predicted labels
print(predicted_labels)



# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# (1000,10,1)
# (1000,1)
# # Generate dummy training data
# X_train = np.random.random((1000, 10, 1))  # Input sequences of shape (batch_size, sequence_length, input_dim)
# y_train = np.random.randint(2, size=(1000, 1))  # Target labels (0 or 1)
#
# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(64, input_shape=(10, 1)))  # 64 is the number of LSTM units
# model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification
#
# # Compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train, y_train, epochs=1, batch_size=32)
#
# # Generate dummy test data
# X_test = np.random.random((100, 10, 1))
# y_test = np.random.randint(2, size=(100, 1))
#
# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
#
# print("Test loss:", loss)
# print("Test accuracy:", accuracy)
