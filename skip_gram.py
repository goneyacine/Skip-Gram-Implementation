import zipfile 
import collections
import numpy as np
import tensorflow as tf
import os
import wandb
class model:


 def __init__(self,max_vocabulary_size=5000,min_occurrence=10,skip_window=3,vector_dim=50,learning_rate=0.0001):
       self.max_vocabulary_size = max_vocabulary_size
       self.min_occurrence = min_occurrence
       self.skip_window = skip_window
       self.vector_dim = vector_dim
       self.learning_rate = learning_rate
       self.load_data()
       self.init_model(vector_dim=vector_dim)
       self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

 def load_data(self):
     data_path = 'text8_dataset/text8.zip'
     with zipfile.ZipFile(data_path) as f:
         text_words = f.read(f.namelist()[0]).lower().split()
           # Build the dictionary and replace rare words with UNK token.
     text_words= text_words[:1000000]
     self.count = [('UNK', -1)]
     self.count.extend(collections.Counter(
         text_words).most_common(self.max_vocabulary_size - 1))
     
     # Remove samples with less than 'min_occurrence' occurrences.
     for i in range(len(self.count) - 1, -1, -1):
         if self.count[i][1] < self.min_occurrence:
             self.count.pop(i)
     # Compute the vocabulary size.
     self.vocabulary_size = len(self.count)
     
     # Assign an id to each word.
     word2id = dict()
     for i, (word, _) in enumerate(self.count):
         word2id[word] = i
     
     data = list()
     unk_count = 0
     for word in text_words:
         # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary.
         index = word2id.get(word, 0)
         if index == 0:
             unk_count += 1
         data.append(index)
     self.count[0] = ('UNK', unk_count)
     self.id2word = dict(zip(word2id.values(), word2id.keys()))
     #remove UNK words from the data
     self.data = [x for x in data if x != 0]

     print("total words count is :" + str(len(data)))
     print("vocabulary size is : " + str(self.vocabulary_size))
     
 def next_batch(self,batch_size ,skip_window=3):
    if(self.data_index + batch_size + skip_window) >= len(self.data):
        return np.ndarray(shape=(0)),np.ndarray(shape=(0,0))
    center_words = np.ndarray(shape=(batch_size,self.vocabulary_size), dtype=np.int32)
    context_words = np.ndarray(shape=(batch_size,self.vocabulary_size,skip_window * 2), dtype=np.int32)
    for i in range(batch_size):
        center_words[i] = self.id_to_one_hot(self.data[self.data_index + i])
        context = np.ndarray(shape=( self.vocabulary_size,skip_window * 2))
        for j in range(skip_window*2):
            if j - skip_window != 0:
               context[:,j] = self.id_to_one_hot(self.data[self.data_index + i + j - skip_window]) 
        context_words[i] = context
    self.data_index += batch_size + skip_window
    return center_words, context_words

 def init_model(self,vector_dim=50):
    input = tf.keras.Input(shape=(self.vocabulary_size))
    projection = tf.keras.layers.Embedding(self.vocabulary_size,vector_dim)(input)
    output =  tf.keras.layers.Dense(self.skip_window * 2,activation='softmax')(projection)
    self.model = tf.keras.models.Model(inputs=input,outputs=output)
    

 #TO DO : Don't forget to connect to the weights and biases api
 def train(self,epochs=3,batch_size=128,auto_save=True):
  wandb.init(
       project="Skip-Gram-Implementation",
       config={
           'vector-dim':self.vector_dim,
           'learning-rate':self.learning_rate,
           'skip-window':self.skip_window,
           'vocabulary-size':self.vocabulary_size,
           'min_occurrence':self.min_occurrence,
           'optimizer':'Adam',
           'batch_size':batch_size,
           'epochs':epochs,
           'total-batches-count':int(len(self.data) / batch_size)
           }
   )
  @tf.function          
  def step(x,y):
    with tf.GradientTape() as tape:
       y_pred = self.model(x)
       loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, y_pred + 0.001, from_logits=False))
    gradients = tape.gradient(loss,
                                self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients,
                                            self.model.trainable_variables))
    return loss
  for i in range(epochs):
      self.data_index = self.skip_window + 1
      print(f'epoch : {i+1}')
      average_loss = 0
      for j in range(int(len(self.data) / batch_size)):
            center_words,context_words = self.next_batch(batch_size=batch_size)
            if np.size(center_words) != 0 and np.size(context_words) != 0:
             step_loss = step(center_words,context_words).numpy()
             wandb.log({'loss':step_loss})
             average_loss +=  step_loss / int(len(self.data) / batch_size)
            if auto_save:
             self.save() 
      print(f'average_loss :{average_loss}')
      wandb.log({'average_loss': average_loss})
 def evaluate(self):
     pass 
 def save(self,output_path='model'):
       if not os.path.exists(output_path):
            os.mkdir(output_path)
       self.model.save(os.path.join(output_path,'skip_gram.keras'))
 def load(self,folder_path='model'):
     self.model = tf.keras.models.load_model(os.path.join(folder_path,'skip_gram.keras'))
  
 
 def id_to_one_hot(self,id):
     one_hot = np.zeros(self.vocabulary_size)
     one_hot[id] = 1
     return one_hot






