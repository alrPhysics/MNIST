# Training a CNN to Read Digits From MNIST

The architecture of the CNN is currently based on the CNN from this [mini course](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0) and the Image Classification project completed as part of Udacity's MLND.

The data is passed through 3 convolutional layers, flattened, and then passed through a fully connected layer and finally the output layer.

As per the above mentioned mini course no pooling layers were implemeneted. Instead a stride of 2 was used for the 2nd and 3rd convolutional layers.

## Possible Future Improvements, Experiments, etc.:

* Initialize bias using `tf.truncated_normal`
* Practice using a dilated convolutional layer, although this may be better suited to a data set with larger images
* Otherwise alter the architecture of the neural network
