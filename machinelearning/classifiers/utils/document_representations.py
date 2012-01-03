'''
Created on Jan 1, 2012

@author: Madan Thangavelu
'''
import re

class BagOfWords(object):
    def __init__(self, tokens = None):
        self._words = {}
        self._no_documents = 0
        if tokens:
            self.add_tokens(tokens)
        
    def add_tokens(self, tokens):
        self._no_documents += 1
        for token in tokens:
            try:
                self._words[token] += 1
            except KeyError:
                self._words[token] = 1
    
    def word_count(self, word):
        try:
            return self._words[word]
        except KeyError:
            return 0
    
class Documents(object):
    def __init__(self):
        self._documents = {}
        self._class_occurrence_count = BagOfWords()
        
    def _process_text(self, text):
        ''' 
        Tokenize, stem, spell correction, word grouping can be 
        performed in this function. The return text is considered
        the contents of a single document
        '''
        text = re.sub(r'[ ]+', ' ', text)
        tokens = text.split(' ')
        return tokens
    
    def _process_tokens(self, tokens):
        ''' Control whether the inverted document index counts can be
        repeated or should be counted only once 
        e.g., if word1 occurs 5 times in document1 should 
        the word be counted as occurring 5 times or just 1 time.
        '''
        return set(tokens)
    
    def add_training(self, text, class_id):
        tokens = self._process_text(text)
        try:
            self._documents[class_id].add_tokens(tokens)
        except KeyError:
            self._documents[class_id] = BagOfWords(tokens=tokens)
        
        self._class_occurrence_count.add_tokens(self._process_tokens(tokens)) # Count once per class
    
    def word_count_given_class(self, word, class_id):
        ''' 
        Returns the total number of time a given word occurs in a given class
        '''
        try:
            return self._documents[class_id].word_count(word)
        except KeyError:
            return 0
    
    def class_count_given_word(self, word):
        '''
        Returns the number of classes that a given word occurs in
        '''
        return self._class_occurrence_count.word_count(word)
        
        
        
            