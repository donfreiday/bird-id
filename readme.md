#Bird-ID

###Classify images of birds by species using Keras.

###Project Goals

1. Develop a model which can classify bird species by image.
2. Feed the model live, motion detected images from my birdfeeder.

###Notes

This project is my first real foray into machine learning, so many of these notes will no doubt be blindingly obvious to the informed reader. There are many gross simplifications present; please let me know of any mistakes.

#####Terms
**AI** is artificial intelligence, possessed by machines which can perceive their environment and use this information to take actions towards a goal.

**Machine learning** ('ML') is a subfield of AI using statistics to allow computers to "learn" from data.

**Neural networks** ('NN') are ML models, somewhat similar to biological neural networks in our brains, which consist of connnected layers of nodes ('neurons') which can receive input, change their state, and produce output. These models can be trained with labeled data and applied to unlabeled data to discover patterns.

**Deep learning** is jargon for a neural network with many hidden layers.

**Training**, as it applies to a NN, means feeding it labeled data, then having it make predictions about unlabeled data. Accurate predictions result in reinforced network parameters, while bad predictions are disincentivized.

A **Convolutional Neural Network** ('CNN') connects nodes to pixels (or other grouped data features) which are close to one another. They are especially useful for image classification and similar problems.

A **Recurrent Neural Network** ('RNN') has special neurons which can 'remember' prior states and input by connecting to themselves.

**Transfer learning** is transferring the 'knowledge' of a similar, pretrained model to a new model, allowing reuse of network parameters. This means the new model can be trained with a much smaller dataset.

**Keras** is an open source Python NN library, which runs on top of TensorFlow ('TF'), Microsoft Cognitive Toolkit ('CNTK'), or Theano. It provides high-level abstractions for easy and fast prototyping of models, and supports both convolutional and recurrent networks.

**TensorFlow** ('TF') is Google's open-source dataflow programming software library.

###Resources
Trained using the [Caltech-USCD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)  dataset.