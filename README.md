# Cover_type_prediction

Cover type prediction model

There seems to be 2 most important factors in accuracy of the model. Its learning rate 0.01 seemed to be too big, 0.001 provides better results. But the most important factor was size of the network. 
Increasing the size of first hidden layer is the most important factor. It gives the network flexibility to relate different data in various configurations. Batch_size 64 seems to be big enough especially as we have only 7 classes.
We could probably improve minority class result by oversampling it, but still increasing the size of the network significantly improved result on minority class (class at index 3). Dataset contains 581012 examples, 
15% dedicated to testing the model. 54 features from which 40 were one hot encoded (already in provided dataset) soil type. So in reality 14 features

  		precision    recall  f1-score   support

           0       0.94      0.95      0.94     31937
           1       0.96      0.95      0.95     42349
           2       0.91      0.96      0.93      5362
           3       0.90      0.74      0.81       398
           4       0.83      0.84      0.84      1482
           5       0.90      0.88      0.89      2572
           6       0.95      0.92      0.94      3052

    accuracy                           0.94     87152
   macro avg       0.91      0.89      0.90     87152
weighted avg       0.94      0.94      0.94     87152

    '1st run': 'loss: 0.5399 - accuracy: 0.7659',
    '2nd run': 'loss:0.6537 - accuracy: 0.7229 (batch_size=54, learning_rate0.02)',
    '3rd run': 'loss: 0.3851 - accuracy: 0.8394 (batch_size=54, learning_rate=0.005',
    '4th run': 'loss: 0.3692 - accuracy: 0.8466 (batch_size=64 learning_rate=0.005',
    '5th run': 'loss: 0.3142 - accuracy: 0.8680 (batch_size=64 learning_rate=0.003',
    '6th run': 'loss: 0.3557 - accuracy: 0.8493 (batch_size=64 learning_rate=0.003, first layer 256 removed)',
    '7th run': 'loss: 0.3562 - accuracy: 0.8490 (batch_size=81, learning_rate=0.0015, first layer 256 removed)',
    '8th run': 'loss: 0.3063 - accuracy: 0.8715 (batch_size=72, learning_rate=0.0025, first layer 128 )',
    '9th run': 'loss: 0.3077 - accuracy: 0.8704 (batch_size=96, learning_rate=0.002, first layer 128 )',
    '10th run': 'loss: 0.3133 - accuracy: 0.8688 (batch_size=72, learning_rate=0.0025, first layer 128, 6 hidden )',
    '11th run': 'loss: 0.3053 - accuracy: 0.8719 (batch_size=72, learning_rate=0.0025, first layer 256, 5 hidden )',
    '12th run': 'loss: 0.2702 - accuracy: 0.8866 (batch_size=72, learning_rate=0.0020, first layer 512, 6 hidden )',
    '13th run': 'loss: 0.2611 - accuracy: 0.8906 (batch_size=128, learning_rate=0.0020, first layer 512, 6 hidden )',
    '14th run': 'loss: 0.2565 - accuracy: 0.8928 (batch_size=64, learning_rate=0.0015, first layer 512, 6 hidden )',
    '15th run': 'loss: 0.2558 - accuracy: 0.8919 (batch_size=64, learning_rate=0.0010, first layer 512, 6 hidden )',
    '16th run': 'loss: 0.1274 - accuracy: 0.9482 (batch_size=64, learning_rate=0.001, first layer 1024, 4 hidden)'
