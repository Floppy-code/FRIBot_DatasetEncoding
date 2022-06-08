from ntpath import join
import pickle
import os
import numpy as np

from Encoding.EncoderBase import Encoder
from LexicalEditor.LexicalEditor import LexicalEditor

#Encoder for classifier network dataset
class EncoderClassifier(Encoder):
    def __init__(self):
        super().__init__()
        self.dictionary_counter = 0
        self.word_dictionary = {}
        self.lex_editor = LexicalEditor()
        self.dataset = None

    def encode_feature_set(self, input_file, output_file):
        #Loading dataset questions and appending IDs
        print("[INFO] Loading data", end = '')
        lines = self.load_dataset(input_file)
        print("...Done")
    
        #metadata = [x[0:4] for x in lines]
        questions = [x[5:9] for x in lines]
        #answers = [x[10:12] for x in lines]
        
        print("[INFO] Assigning unique IDs to questions", end = '')
        #Format: [(question, question_intent_id)]
        #Intents will be used in label set.
        questions_with_ids = []
        id_counter = 0
        for question_list in questions:
            for q in question_list:
                if q != '':
                    questions_with_ids.append((q, id_counter))
            id_counter += 1
        print("...Done")

        print("[INFO] Removing punctuation and building dictionary", end = '')
        #Removing punctuation, conversion to base form.
        for i in range(0, len(questions_with_ids)):
            #Processing
            processed = self.lex_editor.process_sentence(questions_with_ids[i][0])
            processed_id = questions_with_ids[i][1]
            questions_with_ids[i] = (processed, processed_id)

            #Appending unique words to dictionary
            for word in processed:
                if (word not in self.word_dictionary):
                    self.word_dictionary[word] = self.dictionary_counter
                    self.dictionary_counter += 1

        print("...Done")

        #Conversion to array of word vectors
        print("[INFO] Sentence vectorization", end = '')
        #Feature set format: [np vector of words, id]
        self.dataset = []
        for q in questions_with_ids:
            self.dataset.append((self.vectorize_sentence(q[0]), q[1]))
        print("...Done")

        print("[INFO] Saving dataset", end = '')
        self.save_dataset(output_file + '_dataset.dat')
        print("...Done")

        print("[INFO] Saving dictionary", end = '')
        self.save_dictionary(output_file + '_dictionary.dat')
        print("...Done")


    #Creates a vector of words for this sentence
    def vectorize_sentence(self, sentence):
        arr = np.zeros(len(self.word_dictionary))
        for word in sentence:
            arr[self.word_dictionary[word]] = 1.0

        return arr

    def load_dataset(self, filename):
        f = open(filename, encoding="utf8", mode = 'r')

        lines = []
        for line in f.readlines():
            lines.append(line.rstrip('\n').split(';'))

        return lines[1:]


    def save_dataset(self, filename):
        file = open(filename, 'wb')
        pickle.dump(self.dataset, file)


    def save_dictionary(self, filename):
        file = open(filename, 'wb')
        pickle.dump(self.word_dictionary, file)



#Encoder for LSTM network dataset
class EncoderLSTM(Encoder):
    #TODO
   
    def __init__(self):
        super().__init__()