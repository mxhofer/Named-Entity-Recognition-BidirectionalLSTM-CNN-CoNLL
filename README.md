# Named-Entity-Recognition-BLSTM-CNN-CoNLL
  Keras implementation of the Bidirectional LSTM and CNN model by Chiu and Nichols (2016) for CoNLL news data. See paper: https://arxiv.org/abs/1511.08308. Code adapted from: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs

The implementation differs from the original paper in these ways:
  1) no lexicons/dictionaries
  2) Nadam optimizer used instead of SGD
  3) Parameters: LSTM cell size of 200 (vs 275), dropout of 0.5 (vs 0.68)

These are the original parameter values from Chiu and Nichols (2016):
  EPOCHS = 80               
  DROPOUT = 0.68            
  DROPOUT_RECURRENT = 0.25  # not specified in paper
  LSTM_STATE_SIZE = 275     
  CONV_SIZE = 3             
  LEARNING_RATE = 0.0105    
  OPTIMIZER = Nadam()       # paper uses SGD()

# Result 
  The implementation achieves a test F1 score of ~86 with 30 epochs. Increase the number of epochs to 80 reach an F1 over 90. The score produced in Chiu and Nichols (2016) is 91.62. 

# Dataset
  CoNLL-2003 newswire articles: https://www.clips.uantwerpen.be/conll2003/ner/

  The implementation uses GloVe vector representation from Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. See https://nlp.stanford.edu/projects/glove/

# Run script

  Run jupyter notebook
 ```nn_CoNLL.ipynb
 ```
# Dependencies 
    1) numpy 
    2) Keras
    3) Tensorflow
 
 
 
 
 
