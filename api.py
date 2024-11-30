#!/usr/bin/python3

import re
import numpy as np

words = [
    "make",
    "address",
    "all",
    "3d",
    "our",
    "over",
    "remove",
    "internet",
    "order",
    "mail",
    "receive",
    "will",
    "people",
    "report",
    "addresses",
    "free",
    "business",
    "email",
    "you",
    "credit",
    "your",
    "font",
    "000",
    "money",
    "hp",
    "hpl",
    "george",
    "650",
    "lab",
    "labs",
    "telnet",
    "857",
    "data",
    "415",
    "85",
    "technology",
    "1999",
    "parts",
    "pm",
    "direct",
    "cs",
    "meeting",
    "original",
    "project",
    "re",
    "edu",
    "table",
    "conference",
]

words_french = [ "faire", "adresse", "tout", "3d", "notre", "plus", "supprimer", "internet", "ordre", "courrier", "recevoir", "sera", "gens", "rapport", "adresses", "gratuit", "affaires", "email", "vous", "crédit", "votre", "police", "000", "argent", "hp", "hpl", "george", "650", "laboratoire", "laboratoires", "telnet", "857", "données", "415", "85", "technologie", "1999", "pièces", "pm", "direct", "cs", "réunion", "original", "projet", "ré", "edu", "table", "conférence"]

characters = [";", "(", "[", "!", "$", "#"]

def detect_language(text: str) -> str:
    """
    Detect language of text
    """
    if any(word in text for word in words_french):
        return "fr"
    return "en"
    
def extract(email: str, def_words=None) -> np.array:
    """
    Predict if email is spam or not
    """
    if def_words is None:
        if detect_language(email) == "fr":
            features = extract_features(email, words_french)
        else:
            features = extract_features(email, words)
    else:
        features = extract_features(email, def_words)
    features = np.array(features).reshape(1, -1)
    return features


def extract_features(email, words) -> np.array:
    """
    Extract features from email
    48 attributes of frequency of words = 100 * (number of times word appears in email) / total number of words in email
    6 attributes of type freq of chars
    1 capital_run_length_average attributes average length of uninterrupted sequences of capital letters
    1 capital_run_length_longest length of longest uninterrupted sequence of capital letters
    1 capital_run_length_total | = sum of length of uninterrupted sequences of capital letters | total number of capital letters in the e-mail
    1 nominal {0,1} spam (1) or not (0),
    """
    count_words = len(email.split())
    count_chr = len(email)
    features_vector = np.zeros(57)

    # 48 Features
    for i, word in enumerate(words):
        features_vector[i] = 100 * (email.count(word) / count_words)

    # 6 Features
    for i, char in enumerate(characters, start=len(words)):
        features_vector[i] = 100 * (email.count(char) / count_chr)

    # Capitals
    capitals = re.findall(r"[A-Z]+", email)
    i = len(words) + len(characters)
    if capitals:
        capitals_len = [len(cap) for cap in capitals]
        capitals_sum_len = sum(capitals_len)
        capitals_max_len = max(capitals_len)
        capitals_mean_len = capitals_sum_len / len(capitals_len)

        features_vector[i] = capitals_mean_len
        features_vector[i + 1] = capitals_max_len
        features_vector[i + 2] = capitals_sum_len
    else:
        features_vector[i] = 0
        features_vector[i + 1] = 0
        features_vector[i + 2] = 0

    return features_vector
