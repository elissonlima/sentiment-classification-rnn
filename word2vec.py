import re
from turtle import forward
import numpy as np

def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)


class EmbeddingModel:

    def __init__(self, vocab_size, n_embedding):
        
        self.W1 = np.random.randn(vocab_size, n_embedding)
        self.W2 = np.random.randn(n_embedding, vocab_size)

    def forward(self, X, return_layer_values=True):

        layer_values = {}
        layer_values['a1'] = X @ self.W1
        layer_values['a2'] = layer_values['a1'] @ self.W2
        layer_values["z"]  = softmax(layer_values['a2'])

        if not return_layer_values:
            return layer_values["z"]
        
        return layer_values
    
    def backpropagation(self, X, y, alpha):
        layer_values = self.forward(X)
        da2 = layer_values["z"] - y
        dw2 = layer_values['a1'].T @ da2
        da1 = da2 @ self.W2.T
        dw1 = X.T  @ da1

        assert(dw2.shape == self.W2.shape)
        assert(dw1.shape == self.W1.shape)

        self.W1 -= alpha * dw1
        self.W2 -= alpha * dw2

        return cross_entropy(layer_values['z'], y)

    def fit(self, X, y, epochs, lr=0.05):

        history = []
        for _ in range(epochs):
            loss = self.backpropagation(X, y, lr)
            history.append(loss)
        
        # import matplotlib.pyplot as plt
        # plt.style.use("seaborn")

        # plt.plot(range(len(history)), history, color="skyblue")
        # plt.show()

class Embedding:

    def _tokenize(self, text):
        '''
            It splits the text up into smaller units like words
            Parameters: 
                text : str
                    The text that we should extract the tokens
                        Returns:
                tokens : list
                    List of tokens
        '''
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        return pattern.findall(text.lower())

    def _mapping(self, tokens):
        '''
            It generates a map between tokens and indices.
            Parameters:
                tokens : list
                    List of tokens
            Returns:
                word_to_id : dict
                    Map from token to indices
                id_to_word : dict
                    Map from indices to tokens
        '''
        word_to_id = {}
        id_to_word = {}

        for i, token in enumerate(set(tokens)):
            word_to_id[token] = i
            id_to_word[i] = token

        return word_to_id, id_to_word    

    def _one_hot_encode(self, id, vocab_size):
        '''
            It generates on hot encode representation for a token's indice
            Parameters:
                id : int
                    Indice of a specific token
                vocab_size : int
                    The amount of distincts tokens
            Returns:
                one_hot_repr : int
                    One hot representation for a token
        '''
        res = [0] * vocab_size
        res[id] = 1
        return res

    def _generate_training_data(self, window=2):
        '''
            It generates a training set that fits the embedding model. 
            Parameters:
                window : int, optional
                    The size of window considered in the context
            Returns:
                X : np.array
                    The train examples
                y : np.array
                    The train labels
        '''

        # I have no idea what this function does
        def concat(*iterables):
            for iterable in iterables:
                yield from iterable


        X = []
        y = []
        n_tokens = len(self.tokens)

        for i in range(n_tokens):
            idx = concat(
                range(max(0, i - window), i), 
                range(i, min(n_tokens, i + window + 1))
            )
            for j in idx:
                if i == j:
                    continue
                
                X.append(self._one_hot_encode(self.word_to_id[self.tokens[i]], 
                                        len(self.word_to_id)))
                y.append(self._one_hot_encode(self.word_to_id[self.tokens[j]], 
                                        len(self.word_to_id)))    
        
        return np.asarray(X), np.asarray(y)    

    def __init__(self, text, n_embedding):

        self.tokens = self._tokenize(text)
        self.word_to_id, self.id_to_word = self._mapping(self.tokens)

        self.model = EmbeddingModel(len(self.word_to_id), n_embedding)
        X, y = self._generate_training_data()
        self.model.fit(X,y, 50)

    
    def get_context_words(self, word, count=None):
        _word = self._one_hot_encode(self.word_to_id[word], len(self.word_to_id))
        result = self.model.forward([_word], return_layer_values=False)[0]

        word_list = []
        for word in (self.id_to_word[id] for id in np.argsort(result)[::-1]):
            word_list.append(word)
        
        if count is not None:
            return word_list[:count]
        return word_list

    def __call__(self, word):
        
        try:
            idx = self.word_to_id[word]
        except KeyError:
            print("`word` not in corpus")
        
        one_hot = self._one_hot_encode(idx, len(self.word_to_id))
        
        return self.model.forward(one_hot)["a1"]
        


if __name__ == '__main__':

    text = '''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''

    embedding = Embedding(text, 10)
    print(embedding("learning"))