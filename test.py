import json
from collections import defaultdict
import pandas as pd
import numpy as np

START_TAG = '<$S>'
END_TAG = '<$E>'
POS = {START_TAG, 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X', '.', END_TAG}


def generate_transition_matrix(tagged_sentences):
    transition_matrix = {p: defaultdict(int) for p in POS}

    for tagged_sentence in tagged_sentences:
        for i in range(len(tagged_sentence) - 1):
            curr_pos = START_TAG if i == 0 else tagged_sentence[i][1]
            next_pos = END_TAG if i == len(tagged_sentence) - 1 else (
                tagged_sentence[i][1] if curr_pos == START_TAG else tagged_sentence[i + 1][1])
            transition_matrix[curr_pos][next_pos] += 1

    # Normalize it by dividing over the sum of values of the transitions for each POS
    for pos, transition_counts in transition_matrix.items():
        count_sum = sum(transition_counts.values())
        for next_pos in transition_counts:
            transition_counts[next_pos] = transition_counts[next_pos] / count_sum

    transition_df = pd.DataFrame(transition_matrix).fillna(0).T
    transition_df['prob'] = transition_df.sum(axis=1)
    return transition_df


def generate_emission_matrix(tagged_sentences):
    emission_matrix = {}
    for tagged_sentence in tagged_sentences:
        for i in range(len(tagged_sentence) - 1):
            curr_word = tagged_sentence[i][0].lower()
            curr_pos = tagged_sentence[i][1]
            if curr_word not in emission_matrix:
                emission_matrix[curr_word] = {}
                emission_matrix[curr_word][curr_pos] = 1
            else:
                if curr_pos not in emission_matrix[curr_word]:
                    emission_matrix[curr_word].update({curr_pos: 1})
                else:
                    emission_matrix[curr_word][curr_pos] += 1
    emission_df = pd.DataFrame(emission_matrix).fillna(0).T

    for col in emission_df.columns:
        emission_df[col] = emission_df[col] / emission_df[col].sum()

    return emission_df


with open('bc.train', 'r') as train_data:
    data = train_data.readlines()
    tagged_sentences = []

    # Generate (word,POS) pairs from the data for building the transition matrix



    for d in data:
        tagged_sentence = []
        sentence = d.split(' ')
        for idx, word in enumerate(sentence):
            if word in POS:
                if word == '.' and sentence[idx + 1] in POS:  # . .
                    tagged_sentence.append((word, sentence[idx + 1]))
                    break
                tagged_sentence.append((sentence[idx - 1], word))
        tagged_sentences.append(tagged_sentence)

    edf = generate_emission_matrix(tagged_sentences)
    edf.to_csv('emissions.csv')
    tdf = generate_transition_matrix(tagged_sentences)
    tdf.to_csv('transitions.csv')
