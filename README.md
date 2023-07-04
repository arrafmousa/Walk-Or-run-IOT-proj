# Walk-Or-run-IOT-proj


# to run :
upload the data from the in a directory called dataIOT
run main

# to test:
predictions = model.predict(np.array([data_samples[0]]))
predicted_labels = ['walking' if pred < 0.5 else 'running' for pred in predictions]

Print the predicted labels
print(predicted_labels)
