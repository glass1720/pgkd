from sklearn.metrics import precision_recall_fscore_support

labels = ["pig", "dog", "cat", "dog", "cat", "pig"]
predictions = ["pig", "dog", "cat", "dog", "pig", "dog"]

output = precision_recall_fscore_support(
    labels, predictions, average=None, zero_division=0, labels=["pig", "dog", "cat"]
)

# output is of the form (precision, recall, fscore, support
print(output[0])
print(output[1])
print(output[2])
print(output[3])
