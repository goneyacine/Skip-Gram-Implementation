import zipfile 
import collections
import numpy as np
import tensorflow as tf
import os
import wandb
import json
import time
class model:

# IMPORTANT TODO:  INSTEAD OF UNSING CLIPING TO AVOID EXPLODING GRADIENTS, JUST NORMALIZE THEM GRAD = GRAD / MAX(GRAD)
# ALSO TRY TO CHANGE THE WEIGHTS INITIALIZATION METHOD
# APPLY BATCH NORMALIZATION
 def __init__(self,max_vocabulary_size=5000,min_occurrence=10,skip_window=3,vector_dim=50,learning_rate=0.01
              ,train_split=40,valid_split=5,steps_to_valid=5000):
       """
       Args:
           steps_to_valid (int): number of train steps to validate the model 
       """
       self.max_vocabulary_size = max_vocabulary_size
       self.min_occurrence = min_occurrence
       self.skip_window = skip_window
       self.vector_dim = vector_dim
       self.learning_rate = learning_rate
       self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
       self.train_split = train_split
       self.valid_split = valid_split
       self.steps_to_valid = steps_to_valid

 def load_data(self):
     data_path = 'text8_dataset/text8.zip'
     with zipfile.ZipFile(data_path) as f:
         text_words = f.read(f.namelist()[0]).lower().split()
           # Build the dictionary and replace rare words with UNK token.
     self.count = [('UNK', -1)]
     # Remove samples with less than 'min_occurrence' occurrences.
     for i in range(len(self.count) - 1, -1, -1):
         if self.count[i][1] < self.min_occurrence:
             self.count.pop(i)
     self.count.extend(collections.Counter(
         text_words).most_common(self.max_vocabulary_size - 1))
     
     # Compute the vocabulary size.
     self.vocabulary_size = len(self.count)
     
     # Assign an id to each word.
     self.word2id = dict()
     for i, (word, _) in enumerate(self.count):
         self.word2id[word.decode("utf-8")] = i
     
     data = list()
     unk_count = 0
     for word in text_words:
         # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary.
         index = self.word2id.get(word.decode("utf-8"), 0)
         if index == 0:
             unk_count += 1
         data.append(index)
     self.count[0] = ('UNK', unk_count)
     self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))
     #remove UNK words from the data
     data = [x for x in data if x != 0]
     self.train_data = data[:int(len(data) * self.train_split / 100)]
     self.valid_data = data[int(len(data) * self.train_split / 100): int(len(data) * self.train_split / 100)+ int(len(data)* self.valid_split / 100)]
     print("total words count is :" + str(len(data)))
     print("total train words count is :" + str(len(self.train_data)))
     print("total valid words count is :" + str(len(self.valid_data)))
     print("vocabulary size is : " + str(self.vocabulary_size))
     
 def next_batch(self,data_index,data,batch_size ,skip_window=3):
    if(data_index + batch_size + skip_window) >= len(data):
        return np.ndarray(shape=(0)),np.ndarray(shape=(0,0))
    center_words = np.ndarray(shape=batch_size, dtype=np.int32)
    context_words = np.ndarray(shape=(batch_size,self.vocabulary_size,skip_window * 2), dtype=np.int32)
    for i in range(batch_size):
        #center_words[i] = self.id_to_one_hot(self.data[self.data_index + i])
        center_words[i] = data[data_index + i]
        context = np.ndarray(shape=( self.vocabulary_size,skip_window * 2))
        for j in range(skip_window*2):
            if j - skip_window != 0:
               context[:,j] = self.id_to_one_hot(data[data_index + i + j - skip_window]) 
        context_words[i] = context
    data_index += batch_size + skip_window
    return tf.convert_to_tensor(center_words), tf.convert_to_tensor(np.sum(context_words,axis=2))

 def valid(self,batch_size):
     data_index  = self.skip_window + 1
     batches_count = int(len(self.valid_data) / batch_size)
     loss = 0
     for j in range(batches_count):
         center_words,context_words = self.next_batch(data_index,self.valid_data,batch_size)
         data_index += 1
         embed = tf.nn.embedding_lookup(self.embed_matrix,center_words)
         y_pred = tf.add(tf.matmul(embed, self.embed_weights_matrix), self.embed_biases_matrix)
         loss += tf.keras.losses.BinaryCrossentropy(from_logits=True)(context_words,y_pred) / batches_count
     return loss         
 def safe_divide(self,grad):
    if isinstance(grad, tf.IndexedSlices):
        # Get the maximum value in the IndexedSlices
        max_grad_value = tf.reduce_max(tf.abs(grad.values))
        # Scale only if max value > 1
        return tf.cond(
            max_grad_value > 1,
            lambda: tf.IndexedSlices(grad.values / max_grad_value, grad.indices, grad.dense_shape),
            lambda: grad
        )
    else:
        # Regular Tensor case
        max_grad_value = tf.reduce_max(tf.abs(grad))
        return tf.cond(
            max_grad_value > 1,
            lambda: grad / max_grad_value,
            lambda: grad
        )
 def init_model(self):
    self.embed_matrix = tf.Variable(tf.random.uniform([self.vocabulary_size,self.vector_dim],-1,1),dtype=tf.float32)
    self.embed_weights_matrix = tf.Variable(tf.keras.initializers.GlorotNormal()([self.vector_dim,self.vocabulary_size]),dtype=tf.float32)
    self.embed_biases_matrix = tf.Variable(tf.keras.initializers.GlorotNormal()([self.vocabulary_size]),dtype=tf.float32)

 def train(self,epochs=3,batch_size=128,negative_sampling=False,auto_save=True,steps_to_save=5000):
  config={
          'vector_dim':self.vector_dim,
          'learning_rate':self.learning_rate,
          'skip_window':self.skip_window,
          'vocabulary_size':self.vocabulary_size,
          'min_occurrence':self.min_occurrence,
          'negative_sampling':negative_sampling,
          'optimizer':'Adam',
          'batch_size':batch_size,
          'epochs':epochs,
          'train_words_count':len(self.train_data),
          'valid_words_count':len(self.valid_data),
          'steps_to_valid':self.steps_to_valid,
          }     
  wandb.init(
       project="Skip-Gram-Implementation",
       config=config
   )
  @tf.function          
  def step(x,y):
    with tf.GradientTape() as tape:
                embed = tf.nn.embedding_lookup(self.embed_matrix,x)
                y_pred = tf.add(tf.matmul(embed,self.embed_weights_matrix),self.embed_biases_matrix)
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y,y_pred)
    gradients = tape.gradient(loss, [self.embed_matrix, self.embed_weights_matrix, self.embed_biases_matrix])
    #gradients normalization 
    gradients = [self.safe_divide(grad) for grad in gradients]
    self.optimizer.apply_gradients(zip(gradients, [self.embed_matrix, self.embed_weights_matrix, self.embed_biases_matrix]))
    return loss
  for i in range(epochs):
      data_index = self.skip_window + 1
      print(f'epoch : {i+1}')
      average_loss = 0
      for j in range(int(len(self.train_data) / batch_size)):
            center_words,context_words = self.next_batch(data_index,self.train_data,batch_size)
            data_index += 1
            if np.size(center_words) != 0 and np.size(context_words) != 0:
             step_loss = step(center_words,context_words)
             wandb.log({'train_loss':step_loss.numpy()})
             average_loss +=  step_loss / int(len(self.train_data) / batch_size)
             if float(j)%float(self.steps_to_valid) == 0 and j != 0:
                valid_loss = self.valid(batch_size)
                print('valid loss :' + str(valid_loss.numpy()))
                wandb.log({'valid_loss':valid_loss.numpy()})
             if auto_save and float(j)%float(steps_to_save) == 0 and j != 0:
              self.save(config=config) 
      print(f'average_loss :{average_loss}')
      wandb.log({'average_loss': average_loss})
 def save(self,config=None,save_embedings=True,save_model=True,output_path='Experiments'):
       if not os.path.exists(output_path):
            os.mkdir(output_path)
       path = os.path.join(output_path,str(time.time()))
       os.mkdir(path)
       with open(os.path.join(path,'word2id.json'),'w') as word2id_file:
            word2id_file.write(json.dumps(self.word2id))
       if save_embedings:
        np.savetxt(os.path.join(path,'embeding matrix'),self.embed_matrix.numpy())
       if save_model:
        np.savetxt(os.path.join(path,'weights matrix'),self.embed_weights_matrix.numpy())
        np.savetxt(os.path.join(path,'biases matrix'),self.embed_biases_matrix.numpy())
       if not config == None:
        with open(os.path.join(path,'config.txt'),'w') as config_file:
           config_file.write(json.dumps(config))
 def load(self,folder_path,load_embedings=True,load_word2id=True,load_model=False):
     if load_embedings:
        self.embed_matrix = tf.convert_to_tensor(np.load(os.path.join(folder_path,'embeding matrix')))
     if load_word2id:
        with open(os.path.join(folder_path,'word2id.json'),'r') as word2id_file:
             self.word2id = json.load(word2id_file)
     if load_model:
         self.embed_weights_matrix = tf.convert_to_tensor(np.load(os.path.join(folder_path,'weights matrix')))
         self.embed_biases_matrix = tf.convert_to_tensor(np.load(os.path.join(folder_path,'biases matrix')))
 
 def id_to_one_hot(self,id):
     one_hot = np.zeros(self.vocabulary_size)
     one_hot[id] = 1
     return one_hot
 
 def word2vec(self,word):
     word_id = self.word2id[word]
     return self.embed_matrix.numpy()[word_id]
     






