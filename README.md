# Deep Learning With MNIST
Currently this repository contains projects using a convolutional neural network (CNN) and a deep convolutional generative adversarial network (DCGAN). The CNN determines the number (between 0-9) of the handwritten digit and the DCGAN produces original intepretations of handwritten digits.

## Training a CNN to Read Digits From MNIST

The architecture of the CNN is currently based on the CNN from this [mini course](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0) and the [Image Classification](https://github.com/alrPhysics/MachineLearningEngineerNanodegree/tree/master/image_classification) project completed as part of Udacity's Machine Learning Engineer Nanodegree. Similar to the Image Classification project, nothing from tf.layers was implemented in order to practice using the lower level modules.

The data is passed through 3 convolutional layers, flattened, and then passed through a fully connected layer and finally the output layer.

As per the above mentioned mini course no pooling layers were implemeneted. Instead a stride of 2 was used for the 2nd and 3rd convolutional layers.

### Possible Future Improvements, Experiments, etc.:

* Initialize bias using `tf.truncated_normal`
* Practice using a dilated convolutional layer, although this may be better suited to a data set with larger images
* Otherwise alter the architecture of the neural network

## Training a DCGAN to Generate Original Digits

The architecure of the DCGAN is based on this [paper](https://arxiv.org/abs/1511.06434) and this [blog post](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0), but is not as deep as either of these. I was trying to keep it as shallow as possible, while still generating decent results, in order to reduce the amount of time my laptop spent training. The DCGAN easily took ~45 minutes to generate the results in the notebook (I am trying to avoid running my laptop at full tilt for hours for one training session). This was built nearly exclusively with tf.layers.


### Generator:
Upsamples a vector of 100 random numbers to an 'image' that is 28x28 and hopefully looks like a handwritten digit. Tries to get the discriminator to accept its generated image as real.
#### Structure
* dense(x, units = 7*7*64)
* reshape(x, \[-1,7,7,64\])       
* conv2d_transpose(x, filters=64, kernel_size=3, strides=1, padding='same',activation=lrelu)
* conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='same')
* conv2d_transpose(x, filters=32, kernel_size=5, strides=2, padding='same', activation=lrelu)
* conv2d_transpose(x, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
* conv2d_transpose(x, filters=1, kernel_size=3, strides=1, padding='same', activation=tf.nn.tanh)

### Discriminator:
Compares generator output to digits from MNIST and attempts to consistently rate the generator output as fake.
#### Structure
* conv2d(x, filters=32, kernel_size=3, strides=1, padding='same')
* conv2d(x, filters=64, kernel_size=5, strides=2, padding='same')
* conv2d(x, filters=64, kernel_size=5, strides=2, padding='same',activation=lrelu)
* conv2d(x, filters=128, kernel_size=3, strides=1, padding='same',activation=lrelu)
* flatten(x)
* dense(x,units=100)
* dense(x,units=2,activation=tf.nn.tanh)

### Optimizer
The AdamOptimizer was used to minimize the discriminator and generator loss functions.
* Discriminator: learning_rate = 0.0002, beta1 = 0.4
* Generator: learning_rate = 0.00033, beat1 = 0.55

### Notes:
* This architecutre may change and will be updated here if I am able to produce better results.
