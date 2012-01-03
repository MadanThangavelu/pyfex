'''
Created on Dec 31, 2011

@author: Madan Thangavelu

An implementation of "A Class-Feature-Centroid Classifier for Text Categorization"
http://www.google.com/url?sa=t&rct=j&q=a%20class%20feature%20centroid&source=web&cd=1&ved=0CCEQFjAA&url=http%3A%2F%2Fwww2009.eprints.org%2F21%2F1%2Fp201.pdf&ei=VxEAT8XkHPLciQKO-rHiDg&usg=AFQjCNHG3KOMSLtP5emO1Qk6p1c8bpsAHw
'''

from classifiers.utils.document_representations import Documents, BagOfWords
from numpy import log

class ClassCentroidClassifier(Documents):
    def __init__(self):
        super(ClassCentroidClassifier, self).__init__()
        self._class_centroids = {}
    
    def _get_testing_data(self):
        ''' A generator for test strings '''
        testing_data = ['Madan is', 'What is your name']
        class_labels   = [1, 2]
        for testing_text, class_label in zip(testing_data,  class_labels):
            yield (testing_text, class_label)
    
    def _get_training_data(self):
        ''' 
        Return an iterator or a list 
        containing document text
        '''
        training_data = ['Madan is here', 'What is your name', 'google is a name', 'Madan were here']
        class_labels   = [1, 2, 3, 1]
        return zip(training_data, class_labels)
    
    def calculate_class_centroids(self):
        ''' Calculating class centroids '''
        for class_label in self._documents.keys():
            bag_of_words_for_class_label = self._documents[class_label]
            centroid = {}
            for word in bag_of_words_for_class_label._words.keys():
                word_weight = pow(1.7, (1.0*bag_of_words_for_class_label.word_count(word)/bag_of_words_for_class_label._no_documents))*log(self._class_occurrence_count._no_documents/self._class_occurrence_count.word_count(word))                
                centroid[word] = word_weight
            self._class_centroids[class_label] = centroid
    
    def predict(self, test_dict):
        from numpy import Inf
        closest_centroid_class = 0
        predicted_classes = []
        for class_label, centroid in self._class_centroids.items():
            denormalied_cosine_similarity = 0
            for word in centroid.keys():
                denormalied_cosine_similarity += test_dict.get(word,0)*centroid.get(word, 0)
            if denormalied_cosine_similarity > closest_centroid_class:
                predicted_classes = [class_label]
                closest_centroid_class = denormalied_cosine_similarity
            elif denormalied_cosine_similarity == closest_centroid_class:
                predicted_classes.append(class_label)
                closest_centroid_class = denormalied_cosine_similarity
        return predicted_classes
    
    def train(self):
        # Add training data
        for training_sample, class_label in self._get_training_data():
            self.add_training(training_sample, class_label)
        
        # Train system by calculating centroids
        self.calculate_class_centroids()
        
    def test(self):
        misclassified_count  = 0
        total_tests          = 0
        multiple_predictions = 0
        
        for testing_text, class_label in self._get_testing_data():
            total_tests += 1
            tokens = self._process_text(testing_text)
            tokens = self._process_tokens(tokens)
            testing_dict = {}
            word_length = len(tokens)
            for token in tokens:                
                tf = 1.0/word_length
                idf = log(self._class_occurrence_count._no_documents/self._class_occurrence_count.word_count(token))
                testing_dict[token] = tf*idf
            predicted_class_labels = self.predict(testing_dict)
            
            if class_label and (class_label not in predicted_class_labels):
                misclassified_count += 1
            if len(predicted_class_labels) > 1:
                multiple_predictions += 1
        
        print ("Total test data", total_tests)
        print ("Total Failures", misclassified_count)
        print ("Multiple Predictions", multiple_predictions)
        
        return total_tests, misclassified_count, multiple_predictions
        
if __name__ == "__main__":
    cfc = ClassCentroidClassifier()
    cfc.train()
    cfc.test()
        
            
    
    