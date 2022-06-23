from Encoding.Encoders import EncoderClassifier
from Vectorization.Vectorizer import Vectorizer

#Encoding - Encoding the whole dataset into one-hot vectors and output vectors

encoder = EncoderClassifier()
encoder.encode_feature_set('./data/komplet2.csv', './encoded/komplet2_encoded')


# Vectorization - Turning any sentence into one-hot encoded vector

# vectorizer = Vectorizer('./encoded/otazky_encoded_dictionary.dat')
# while True:
#     input_sentence = input()
#     if (input_sentence == 'e'):
#         break

#     vector = vectorizer.vectorize_sentence(input_sentence)
#     print(vector)
#     print(vectorizer.resentence(vector))