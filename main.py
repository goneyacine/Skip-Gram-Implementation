import skip_gram
import tensorflow as tf
print(tf.version.VERSION)
_model = skip_gram.model(learning_rate=1e-4,vector_dim=300,train_split=60,valid_split=2,max_vocabulary_size=5000)
_model.load_data()
_model.init_model()
_model.train(epochs=1,batch_size=64)


