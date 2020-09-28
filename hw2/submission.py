#!/usr/bin/python

import math
import random
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    word_features: FeatureVector = {}
    for word in x.split(" "):
        word_features[word] = 1 + word_features.get(word, 0)
    return word_features
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]], featureExtractor: Callable[[str], FeatureVector], numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for epoch in range(numEpochs):
        for example in trainExamples:
            example_x, example_y = example
            phi = featureExtractor(example_x)
            gradient_loss: featureExtractor = {}
            if 1 - dotProduct(weights, phi) * example_y > 0:
                increment(gradient_loss, -1 * example_y, phi)
            increment(weights, -1 * eta, gradient_loss)
        predictor = lambda x: 1 if dotProduct(weights, featureExtractor(x)) >= 0 else -1
        print(f"Epoch: {epoch}; train error: {evaluatePredictor(trainExamples, predictor)}; validation error: {evaluatePredictor(validationExamples, predictor)}")
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.

    # Note that the weight vector can be arbitrary during testing. 
    def generateExample() -> Example:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = { random.choice(list(weights.keys())): random.uniform(1, 10) for _ in range(random.randint(0, len(weights))) }
        y = 1 if dotProduct(weights , weights) >= 0 else -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        phi: FeatureVector = {}
        x_chars = x.replace(" ", "")
        for i in range(len(x_chars) - n + 1):
            ngram_str = x_chars[i:i + n]
            phi[ngram_str] = 1 + phi.get(ngram_str, 0)
        return phi
        # END_YOUR_CODE
    return extract

############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

############################################################
# Problem 4: k-means
############################################################

import time

def kmeans(examples: List[Dict[str, float]], K: int, maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    centroids: List[Dict[str, float]] = [random.choice(examples) for k in range(K)]
    centroids_sq: List[float] = [dotProduct(centroid, centroid) for centroid in centroids]
    examples_sq: List[float] = [dotProduct(example, example) for example in examples]
    def getDistFromCentroid(centroid_i, example_i):
        return examples_sq[example_i] + centroids_sq[centroid_i] - 2 * dotProduct(centroids[centroid_i], examples[example_i])
    assignments: List[int] = [0 for _ in range(len(examples))]
    for _ in range(maxEpochs):
        old_assignments = assignments.copy()
        for i in range(len(assignments)):
            assignments[i] = min(range(len(centroids)), key=lambda c_i: getDistFromCentroid(c_i, i))
        if old_assignments == assignments:
            break
        for k in range(K):
            cluster = [i for i, k_ in filter(lambda x: x[1] == k, enumerate(assignments))]
            centroids[k] = {}
            if len(cluster) > 0:
                cluster_sum = {}
                for example_i in cluster:
                    increment(cluster_sum, 1, examples[example_i])
                increment(centroids[k], (1 / len(cluster)), cluster_sum)
            centroids_sq[k] = dotProduct(centroids[k], centroids[k])
    loss = sum(getDistFromCentroid(assignments[i], i) for i in range(len(examples)))
    return (centroids, assignments, loss)
    # END_YOUR_CODE
