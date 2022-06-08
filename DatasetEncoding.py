from Encoding.Encoders import EncoderClassifier

encoder = EncoderClassifier()
encoder.encode_feature_set('./data/otazky.csv', './encoded/otazky_encoded')