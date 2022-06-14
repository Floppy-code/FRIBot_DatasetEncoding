from Encoding.Encoders import EncoderClassifier
from Vectorization.Vectorizer import Vectorizer

#encoder = EncoderClassifier()
#encoder.encode_feature_set('./data/otazky.csv', './encoded/otazky_encoded')

vectorizer = Vectorizer('./encoded/otazky_encoded_dictionary.dat')
while True:
    input_sentence = input()
    if (input_sentence == 'e'):
        break

    vector = vectorizer.vectorize_sentence(input_sentence)
    print(vector)
    print(vectorizer.resentence(vector))