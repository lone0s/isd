#!/usr/bin/python3

import re
import numpy as np

words = {
    "en": [
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
    ],
    "fr": [
        "faire",
        "adresse",
        "tout",
        "3d",
        "notre",
        "plus",
        "supprimer",
        "internet",
        "ordre",
        "courrier",
        "recevoir",
        "sera",
        "gens",
        "rapport",
        "adresses",
        "gratuit",
        "affaires",
        "email",
        "vous",
        "crédit",
        "votre",
        "police",
        "000",
        "argent",
        "hp",
        "hpl",
        "george",
        "650",
        "laboratoire",
        "laboratoires",
        "telnet",
        "857",
        "données",
        "415",
        "85",
        "technologie",
        "1999",
        "pièces",
        "pm",
        "direct",
        "cs",
        "réunion",
        "original",
        "projet",
        "ré",
        "edu",
        "table",
        "conférence",
    ],
    "es": [
        "hacer",
        "dirección",
        "todo",
        "3d",
        "nuestro",
        "más",
        "eliminar",
        "internet",
        "orden",
        "correo",
        "recibir",
        "será",
        "gente",
        "informe",
        "direcciones",
        "gratis",
        "negocios",
        "correo electrónico",
        "tú",
        "crédito",
        "tu",
        "fuente",
        "000",
        "dinero",
        "hp",
        "hpl",
        "george",
        "650",
        "laboratorio",
        "laboratorios",
        "telnet",
        "857",
        "datos",
        "415",
        "85",
        "tecnología",
        "1999",
        "piezas",
        "pm",
        "directo",
        "cs",
        "reunión",
        "original",
        "proyecto",
        "re",
        "edu",
        "tabla",
        "conferencia",
    ],
    "de": [
        "machen",
        "adresse",
        "alle",
        "3d",
        "unser",
        "über",
        "entfernen",
        "internet",
        "bestellung",
        "post",
        "erhalten",
        "wird",
        "menschen",
        "bericht",
        "adressen",
        "frei",
        "geschäft",
        "email",
        "sie",
        "kredit",
        "ihr",
        "schriftart",
        "000",
        "geld",
        "hp",
        "hpl",
        "george",
        "650",
        "labor",
        "labors",
        "telnet",
        "857",
        "daten",
        "415",
        "85",
        "technologie",
        "1999",
        "teile",
        "pm",
        "direkt",
        "cs",
        "treffen",
        "original",
        "projekt",
        "re",
        "edu",
        "tabelle",
        "konferenz",
    ],
}

characters = [";", "(", "[", "!", "$", "#"]


def detect_language(text: str) -> str:
    """
    Detect language of text
    """
    count_en = 0
    count_fr = 0
    count_es = 0
    count_de = 0

    text = text.lower()

    for word in text.split():
        if word in words["en"]:
            count_en += 1
        if word in words["fr"]:
            count_fr += 1
        if word in words["es"]:
            count_es += 1
        if word in words["de"]:
            count_de += 1

    counts = [count_en, count_fr, count_es, count_de]
    print(f"Counts: {counts}")
    max_count = max(counts)
    if max_count == 0:
        return "en"
    else:
        test = list(words.keys())[counts.index(max_count)]
        print(f"Language detected: {test}")
        return test


def extract(email: str, def_words=None) -> np.array:
    """
    Predict if email is spam or not
    """
    if def_words is None: 
        language = detect_language(email) 
    if language != "en": 
        def_words = words[language] 
    else: 
        def_words = words["en"]
    features = extract_features(email, def_words)
    features = np.array(features).reshape(1, -1)
    print("Features extracted")
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
        word = word.lower()
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
