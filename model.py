import zipfile 
import collections
import numpy as np
#data preparation
max_vocabulary_size = 5000
min_occurrence = 10
skip_window = 3

data_path = 'text8_dataset/text8.zip'
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()
      # Build the dictionary and replace rare words with UNK token.
count = [('UNK', -1)]
count.extend(collections.Counter(
    text_words).most_common(max_vocabulary_size - 1))

# Remove samples with less than 'min_occurrence' occurrences.
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        # The collection is ordered, so stop when 'min_occurrence' is reached.
        break
# Compute the vocabulary size.
vocabulary_size = len(count)

# Assign an id to each word.
word2id = dict()
for i, (word, _) in enumerate(count):
    word2id[word] = i

data = list()
unk_count = 0
for word in text_words:
    # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary.
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count)
id2word = dict(zip(word2id.values(), word2id.keys()))
#remove UNK words from the data
data = [x for x in data if x != 0]

data_index = skip_window + 1

def next_batch(batch_size ,skip_window):
    global data_index
    if(data_index + batch_size + skip_window) >= len(data):
        return 
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, skip_window * 2), dtype=np.int32)
    for i in range(batch_size):
        batch[i] = data[data_index + i]
        print('center ' + str(id2word[batch[i]].decode('utf-8')))
        for j in range(skip_window*2):
            if j - skip_window != 0:
               labels[i,j] = data[data_index + i + j - skip_window] 
               print('context ' + str(id2word[labels[i,j]].decode('utf-8')))
    return batch, labels






