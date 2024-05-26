import string
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from fuzzywuzzy import process
from difflib import get_close_matches
from difflib import SequenceMatcher
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# operation on json
def load_json(file_path):
    """
    Load JSON data from a file.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - data (dict or list): The loaded JSON data.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json(file_path, data):
    """
    Save JSON data to a file.

    Parameters:
    - file_path (str): The path to the JSON file.
    - data (dict): The JSON data to be saved.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file)

# preprocessing and functions required
def tokenize(sentence):
    """
    Tokenizes a sentence into words.

    Parameters:
    - sentence (str): The input sentence to tokenize.

    Returns:
    - tokens (list): List of tokens (words) from the input sentence.
    """
    return nltk.word_tokenize(sentence)

def stem(tokens):
    """
    Stems a list of tokens.

    Parameters:
    - tokens (list): List of tokens (words) to stem.

    Returns:
    - stemmed_words (list): List of stemmed tokens.
    """
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return stemmed_words

def get_wordnet_pos(treebank_tag):
    """
    Maps NLTK's POS tags to WordNet's POS tags.

    Parameters:
    - treebank_tag (str): NLTK's POS tag.

    Returns:
    - wordnet_tag (str): WordNet's POS tag.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_tokens(tokens):
    """
    Lemmatizes a list of tokens.

    Parameters:
    - tokens (list): List of tokens to lemmatize.

    Returns:
    - lemmatized_tokens (list): List of lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []

    for token, tag in pos_tag(tokens):
        wordnet_tag = get_wordnet_pos(tag)
        lemmatized_tokens.append(lemmatizer.lemmatize(token, pos=wordnet_tag).lower())
    
    return lemmatized_tokens

def remove_stopwords(words):
    """
    Removes stopwords (punctuation) from a list of words.

    Parameters:
    - words (list): List of words to remove stopwords and punctuation from.

    Returns:
    - filtered_words (list): List of words with stopwords and punctuation removed.
    """
    punctuation = set(string.punctuation)
    return [word for word in words if word not in punctuation]

def preprocessing(input):
    """
    Preprocesses input text by tokenizing, lemmatizing, removing stopwords and merging tokens back into one sentence.

    Parameters:
    - input (str): The input text to preprocess.

    Returns:
    - processedInput (str): Preprocessed input text.
    """
    processedInput = tokenize(input)
    processedInput = lemmatize_tokens(processedInput)
    # processedInput = stem(processedInput) # Uncomment this line if stemming is desired; choose either lemmatize or tokenize
    processedInput = remove_stopwords(processedInput)
    processedInput = " ".join(processedInput)
    return processedInput

# chatbot functions:
# function for chat_bot1
def get_response_extractOne(input, all_words):
    """
    Retrieves the response with the highest similarity using fuzzy string matching (extractOne).

    Parameters:
    - input (str): The input text (after processing) from the user.
    - all_words (list): List of dictionaries containing questions and corresponding answers.

    Returns:
    - answer (str): The response with the highest similarity.
    - score (float): The similarity score of the matched response.
    """
    processedInput = preprocessing(input)
    allquestions = [question['question'].lower() for question in all_words]

    if len(allquestions) == 0:
        return "", 0

    match, score = process.extractOne(processedInput, allquestions)
    idx = allquestions.index(match)
    answer = all_words[idx]['answer']

    # print("input after update: "+processedInput)
    print(f"Similarity between '{input}' and '{all_words[idx]['question']}' is: {score}") 

    return answer, score

# helper function for chat_bot2
def cosine_similarity_sentences(sentence, sentences_list):
    """
    Calculates cosine similarity between a given sentence and a list of sentences.

    Parameters:
    - sentence (str): The first sentence for comparison.
    - sentences_list (list): List of sentences that will be used to compare sentence (1st parameter).

    Returns:
    - similarities (list): List of cosine similarity scores.
    """
    all_sentences = [sentence] + sentences_list
    vectorizer = TfidfVectorizer().fit_transform(all_sentences)
    vectors = vectorizer.toarray()

    sentence1_vector = vectors[0]  # First vector is sentence1 as it's the first in all_sentences
    similarities = []
    for i, vector in enumerate(vectors[1:], start=1):
        similarity = cosine_similarity([sentence1_vector], [vector])
        similarities.append(similarity[0][0])  # cosine_similarity returns a 2D array

    return similarities

# function for chat_bot2
def get_response_cosine(input, all_words):
    """
    Retrieves the response with the highest cosine similarity usingTF-IDF vectors and cosine similarity.

    Parameters:
    - input (str): The input text (after processing) from the user.
    - all_words (list): List of dictionaries containing questions and corresponding answers.

    Returns:
    - answer (str): The response with the highest cosine similarity.
    - highest_similarity (float): The highest cosine similarity score.
    """
    processedInput = preprocessing(input)
    allquestions = [question['question'].lower() for question in all_words]
    results = cosine_similarity_sentences(processedInput, allquestions)

    if len(results) == 0:
        return "", 0

    highest_similarity = max(results)
    idx = results.index(highest_similarity)
    answer = all_words[idx]['answer']

    # print("input after update: "+input)
    print(f"Similarity between '{input}' and '{all_words[idx]['question']}' is: {highest_similarity}") 

    return answer, highest_similarity

# function for chat_bot3
def get_response_close_matches(input, all_words, treshold):
    """
    Retrieves the response with the highest similarity using close matches.

    Parameters:
    - input (str): The input text from the user.
    - all_words (list): List of dictionaries containing questions and corresponding answers.
    - threshold (float): The minimum similarity ratio required for a match.

    Returns:
    - answer (str): The response with the highest similarity.
    - ratio (float): The similarity ratio of the closest match.
    """
    processedInput = preprocessing(input)
    allquestions = [question['question'].lower() for question in all_words]
    matches = get_close_matches(processedInput, allquestions, n=1, cutoff=treshold)

    if len(matches) == 0:
        return "", 0

    idx = allquestions.index(matches[0])
    answer = all_words[idx]['answer']
    ratio = SequenceMatcher(None, processedInput, matches[0]).ratio()

    # print("input after update: "+processedInput)
    print(f"Similarity between '{input}' and '{all_words[idx]['question']}' is: {ratio}") 

    return answer, ratio