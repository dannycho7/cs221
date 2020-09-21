import collections
import math
from queue import Queue
from typing import Any, DefaultDict, List, Set, Tuple

############################################################
# Custom Types
# NOTE: You do not need to modify these.

"""
You can think of the keys of the defaultdict as representing the positions in the sparse vector,
while the values represent the elements at those positions. Any key which is absent from the dict means that
that element in the sparse vector is absent (is zero). Note that the type of the key used should not affect the
algorithm. You can imagine the keys to be integer indices (e.x. 0, 1, 2) in the sparse vectors, but it should work
the same way with arbitrary keys (e.x. "red", "blue", "green").
"""
SparseVector = DefaultDict[Any, float]
Position = Tuple[int, int]


############################################################
# Problem 3a

def find_alphabetically_last_word(text: str) -> str:
    """
    Given a string |text|, return the word in |text| that comes last
    lexicographically (i.e. the word that would come last when sorting).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return max(text.split(" "))
    # END_YOUR_CODE


############################################################
# Problem 3b

def euclidean_distance(loc1: Position, loc2: Position) -> float:
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return math.sqrt(((loc1[0] - loc2[0]) ** 2) + ((loc1[1] - loc2[1]) ** 2))
    # END_YOUR_CODE


############################################################
# Problem 3c

def mutate_sentences(sentence: str) -> List[str]:
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the original sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    sentenceWords = sentence.split(" ")
    wordToAdjWords = collections.defaultdict(set)
    for i, word in enumerate(sentenceWords):
        if i < len(sentenceWords) - 1:
            wordToAdjWords[word].add(sentenceWords[i + 1])
    toProcess = Queue() # [words]
    for word in wordToAdjWords:
        toProcess.put([word])
    answer = []
    while not toProcess.empty():
        words = toProcess.get()
        if len(words) == len(sentenceWords):
            answer.append(" ".join(words))
        else:
            for nextWord in wordToAdjWords[words[-1]]:
                toProcess.put(words + [nextWord])
    return answer
    # END_YOUR_CODE


############################################################
# Problem 3d

def sparse_vector_dot_product(v1: SparseVector, v2: SparseVector) -> float:
    """
    Given two sparse vectors (vectors where most of the elements are zeros) |v1| and |v2|, each
    represented as collections.defaultdict(Any, float), return their dot product.

    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    Note: A sparse vector has most of its entries as 0
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return sum(v1[k] * v2[k] for k in v1)
    # END_YOUR_CODE


############################################################
# Problem 3e

def increment_sparse_vector(v1: SparseVector, scale: float, v2: SparseVector) -> None:
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.

    NOTE: This function should MODIFY v1 in-place, but not return it.
    Do not modify v2 in your implementation.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for k in v2:
        v1[k] += scale * v2[k]
    # END_YOUR_CODE


############################################################
# Problem 3f

def find_singleton_words(text: str) -> Set[str]:
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    wordCounts = collections.defaultdict(int)
    for word in text.split(" "):
        wordCounts[word] += 1
    return set(w for w in filter(lambda w: wordCounts[w] == 1, wordCounts))
    # END_YOUR_CODE


############################################################
# Problem 3g

def compute_longest_palindrome_length(text: str) -> int:
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama' and it's length is 3.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
    if len(text) == 0:
        return 0
    palindrome_lens = [[1 if i <= j else 0 for j in range(len(text))] for i in range(len(text))]
    for l in range(2, len(text) + 1):
        for i in range(len(text) - l + 1):
            j = i + l - 1
            if text[i] == text[j]:
                palindrome_lens[i][j] = 2 + palindrome_lens[i + 1][j - 1]
            else:
                palindrome_lens[i][j] = max(palindrome_lens[i][j - 1], palindrome_lens[i + 1][j])
    return palindrome_lens[0][len(text) - 1]
    # END_YOUR_CODE
