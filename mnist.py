import pickle, gzip, numpy

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
(train_set,train_label), (valid_set, vali_label), (test_set, test_labet )= pickle.load(f, encoding='latin1')
f.close()
