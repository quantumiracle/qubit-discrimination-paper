# qubit-discrimination-paper
This project is for single qubit discrimination on 171Yb^+ trapped-ion system.  

## Methods for this task:
Traditional methods:
* Threshold method------------------------------qubit_threshold.py
* Maximum likelihood method---------------------qubit_max.py
  
Neural Networks (NN) methods:
* Fully-connected Neural Network method------------qubit_fullyNN.py
* Convolutional Neural Network (CNN) method--------qubit_cnn.py
* Recurrent Neural Network (RNN) method------------qubit_rnn.py

Machine Learning classifiers:
* Support Vector Machine (SVM) method-----------------qubit_svm.py
* Logistic Regression--------------------------------\
* K-neighbor Classifier--------------------------------|qubit_clus.py
* Decision Tree Classifier---------------------------/

## To use:
### Neural Networks (NN) methods:

Take *qubit_cnn.py* for example:

`python qubit_cnn.py --train` for training.

`python qubit_cnn.py --test` for testing.

### Other methods:
Just run `python qubit_##.py`to get discrimination results.
