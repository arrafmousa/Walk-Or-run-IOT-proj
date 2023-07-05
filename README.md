# Walk-Or-run-IOT-proj
## to run :
upload the data from the in a directory called dataIOT
run main

## to load pretrained:
model = keras.models.load_model('walk_run_model.mdl')

## to test:
predictions = model.predict(np.array([data_samples[0]]))
predicted_labels = ['walking' if pred < 0.5 else 'running' for pred in predictions]

Print the predicted labels
print(predicted_labels)

# Predict number of steps
## to trian :
run get_threshold([1, 2, 3])
## to eval:
count_entries_above_threshold(arr, entry_idx, thresholds)
where thresholds should e te output of te first stage
entry_idx = [1,2,3]
arr is the data of teh walk you are tying to predict


