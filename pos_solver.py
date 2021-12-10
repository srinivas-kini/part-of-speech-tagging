# References:
# https://github.com/gurjaspalbedi/parts-of-speech-tagging/blob/master/pos_solver.py (ONLY FOR GIBBS SAMPLING)
# https://www.mygreatlearning.com/blog/pos-tagging/ (NO CODE TAKEN)
# https://www.w3schools.com/python/python_regex.asp
# https://dictionary.cambridge.org/us/grammar/british-grammar/suffixes
# Viterbi algorithm given by D. Crandall

from numpy import log10, zeros, argmin, seterr
from pandas import DataFrame
from collections import defaultdict
from string import punctuation
from numpy.random import random
from re import findall

START_TAG = '@@'
POS = [START_TAG, 'noun', 'num', 'pron', 'verb', 'adp', 'det', 'x', '.', 'adj', 'adv', 'prt', 'conj']
DEFAULT_TRANSITION_P = 1 / 12
DEFAULT_EMISSION_P = 0.0000000000001
seterr(divide='ignore', invalid='ignore')


class Solver:

    def __init__(self):
        self.priors = None
        self.transition_matrix = None
        self.emission_matrix = None
        self.complex_transition_matrix = None
        self.complex_emission_matrix = None

    def posterior(self, model, sentence, label):

        priors = self.priors
        e_matrix = self.emission_matrix
        t_matrix = self.transition_matrix
        ct_matrix = self.complex_transition_matrix
        ce_matrix = self.complex_emission_matrix

        log_joint_p = 0
        if model == "Simple":

            for idx, word in enumerate(sentence):
                log_joint_p += log10(priors[label[idx]] * e_matrix.loc[word, label[idx]] if (
                        word in e_matrix.index and e_matrix.loc[word, label[idx]] > 0) else DEFAULT_EMISSION_P)
            return log_joint_p

        elif model == "HMM":
            log_joint_p = log10(priors[label[0]])
            for idx in range(1, len(sentence)):
                log_joint_p += log10(
                    t_matrix.loc[label[idx - 1], label[idx]] * e_matrix.loc[sentence[idx], label[idx]] if (
                            sentence[idx] in e_matrix.index and e_matrix.loc[
                        sentence[idx], label[idx]] > 0) else DEFAULT_EMISSION_P)

            return log_joint_p

        elif model == "Complex":
            for idx, word in enumerate(sentence):
                simple_trans_p = t_matrix.loc[START_TAG, label[idx]] if idx == 0 else t_matrix.loc[
                    label[idx - 1], label[idx]]
                simple_ems_p = e_matrix.loc[word, label[idx]] if (word in e_matrix.index and e_matrix.loc[
                    word, label[idx]] > 0) else DEFAULT_EMISSION_P
                complex_trans_p = DEFAULT_TRANSITION_P
                complex_ems_p = DEFAULT_EMISSION_P

                if idx >= 2:
                    try:
                        complex_trans_p = ct_matrix[label[idx]][label[idx - 1]][label[idx - 2]]
                    except KeyError:
                        pass

                if idx >= 1:
                    try:
                        complex_ems_p = ce_matrix[word][label[idx - 1]][label[idx]]
                    except KeyError:
                        pass
                log_joint_p += log10(simple_trans_p) + log10(complex_trans_p) + log10(simple_ems_p) + log10(
                    complex_ems_p)
            return log_joint_p
        else:
            print("Unknown algo!")

    def train(self, data):
        tagged_sentences = [[(sentence[i], part_of_speech[i]) for i in range(len(sentence))] for
                            sentence, part_of_speech in data]

        self.transition_matrix, self.priors = POSHelper.generate_transition_matrix_and_priors(tagged_sentences)
        self.emission_matrix = POSHelper.generate_emission_matrix(tagged_sentences)

        # For the Bayes net in figure 1(c)
        self.complex_transition_matrix = POSHelper.generate_complex_transition_matrix(tagged_sentences)
        self.complex_emission_matrix = POSHelper.generate_complex_emission_matrix(tagged_sentences)

    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

    def simplified(self, sentence):
        e_matrix = self.emission_matrix
        priors = self.priors

        parts_of_speech = POS[1:]
        most_likely_sequence = []
        # posterior_p = 0

        for word in sentence:
            if word not in e_matrix.index:
                pos = POSHelper.assign_rule_based_tag(word)
                most_likely_sequence.append(pos)
                continue

            max_posterior, most_likely_pos = 0, 'noun'
            for pos in parts_of_speech:
                likelihood = e_matrix.loc[word, pos]
                curr_posterior = (priors[pos] * likelihood)  # Baye's Law (Ignore the evidence for max likelihood)
                if curr_posterior > max_posterior:
                    max_posterior, most_likely_pos = curr_posterior, pos

            # posterior_p += log10(max_posterior)
            most_likely_sequence.append(most_likely_pos)

        # self.simple_posterior = posterior_p
        return most_likely_sequence

    def hmm_viterbi(self, sentence):
        e_matrix = self.emission_matrix
        t_matrix = self.transition_matrix
        priors = self.priors

        parts_of_speech = POS[1:]
        num_labels = len(parts_of_speech)
        num_observations = len(sentence)
        rule_based_tags = [''] * len(sentence)

        viterbi_table = zeros((num_labels, num_observations))
        path_table = zeros((num_labels, num_observations))

        for idx, pos in enumerate(parts_of_speech):
            if sentence[0] in e_matrix.index:
                viterbi_table[idx, 0] = -log10(priors[pos]) + -log10(
                    e_matrix.loc[sentence[0], pos])
            else:
                rule_based_tags[0] = POSHelper.assign_rule_based_tag(sentence[0])
                viterbi_table[idx, 0] = -log10(priors[pos]) + -log10(DEFAULT_EMISSION_P)

        for col in range(1, num_observations):
            for row, pos in enumerate(parts_of_speech):
                path_table[row, col - 1], viterbi_table[row, col] = min(
                    [(i, viterbi_table[i, col - 1] - log10(t_matrix.loc[prev_pos, pos]))
                     for i, prev_pos in enumerate(parts_of_speech)], key=lambda pair: pair[1])

                if sentence[col] in e_matrix.index:
                    viterbi_table[row][col] += -log10(e_matrix.loc[sentence[col], parts_of_speech[row]])
                else:
                    rule_based_tags[col] = POSHelper.assign_rule_based_tag(sentence[col])
                    viterbi_table[row][col] += -log10(DEFAULT_EMISSION_P)

        viterbi_seq = [0 for _ in range(num_observations)]
        viterbi_seq[num_observations - 1] = argmin(viterbi_table[:, -1])

        for i in range(num_observations - 2, -1, -1):
            viterbi_seq[i] = int(path_table[viterbi_seq[i + 1]][i])

        most_likely_sequence = [parts_of_speech[i] for i in viterbi_seq]

        # Replace unseen words with the output of the rule-based tagger
        for i in range(len(rule_based_tags)):
            if rule_based_tags[i]:
                most_likely_sequence[i] = rule_based_tags[i]

        return most_likely_sequence

    def complex_mcmc(self, sentence):
        most_likely_sequence = ['noun'] * len(sentence)
        for _ in range(122):
            most_likely_sequence = self.generate_sample(sentence, most_likely_sequence)
        return most_likely_sequence

    def generate_sample(self, sentence, pos_sample):

        # prior_p = self.priors[pos_sample[0]]
        t_matrix = self.transition_matrix
        e_matrix = self.emission_matrix
        ct_matrix = self.complex_transition_matrix
        ce_matrix = self.complex_emission_matrix

        parts_of_speech = POS[1:]

        for i, word in enumerate(sentence):
            sample_probs = [0 for _ in range(len(parts_of_speech))]
            for j, pos in enumerate(parts_of_speech):
                simple_trans_p = t_matrix.loc[START_TAG, pos] if j == 0 else t_matrix.loc[
                    parts_of_speech[j - 1], pos]
                simple_ems_p = e_matrix.loc[word, pos] if (word in e_matrix.index and e_matrix.loc[
                    word, pos] > 0) else DEFAULT_EMISSION_P
                complex_trans_p = DEFAULT_TRANSITION_P
                complex_ems_p = DEFAULT_EMISSION_P

                if j >= 2:
                    try:
                        complex_trans_p = ct_matrix[pos][parts_of_speech[j - 1]][parts_of_speech[j - 2]]
                    except KeyError:
                        pass

                if j >= 1:
                    try:
                        complex_ems_p = ce_matrix[word][parts_of_speech[j - 1]][pos]
                    except KeyError:
                        pass

                # P(Si-2) * P(Si-1 | Si-2 ) * P(Si | Si-1,Si-2) * P(Wi | Si) * P (Wi | Si-1, Si)
                sample_probs[j] = simple_trans_p * complex_trans_p * simple_ems_p * complex_ems_p

            # Normalize
            sample_probs = [p / sum(sample_probs) for p in sample_probs]
            r = random()
            running_sum = 0

            for k in range(len(sample_probs)):
                running_sum += sample_probs[k]
                if running_sum >= r:
                    pos_sample[i] = parts_of_speech[k]
                    break

        return pos_sample


# Helper class for generating training data and other rule-based heuristics
class POSHelper:

    @staticmethod
    def generate_transition_matrix_and_priors(tagged_sentences):
        transition_dict = {p: defaultdict(int) for p in POS}
        priors = defaultdict(float)
        # P(Qt | Qt-1)
        for tagged_sentence in tagged_sentences:
            for i in range(len(tagged_sentence) - 1):
                curr_partofspeech = START_TAG if i == 0 else tagged_sentence[i][1]
                next_partofspeech = tagged_sentence[i][1] if curr_partofspeech == START_TAG else tagged_sentence[i + 1][
                    1]
                transition_dict[curr_partofspeech][next_partofspeech] += 1
                if curr_partofspeech != START_TAG:
                    priors[curr_partofspeech] += 1.0

        # Normalize over transition probabilities
        transition_dict = POSHelper.normalize(transition_dict)
        transition_matrix = DataFrame(transition_dict).fillna(0).T

        # Normalize over priors
        total_pos_count = sum(priors.values())
        for pos in priors:
            priors[pos] = priors[pos] / total_pos_count

        return transition_matrix, priors

    @staticmethod
    def generate_complex_transition_matrix(tagged_sentences):
        complex_transition_matrix = {p: {p: defaultdict(int) for p in POS[1:]} for p in POS}
        # P(Qt | Qt-1, Qt-2)
        for tagged_sentence in tagged_sentences:
            for i in range(1, len(tagged_sentence)):
                prev_prev_pos, prev_pos, curr_pos = START_TAG if i == 1 else tagged_sentence[i - 2][1], \
                                                    tagged_sentence[i - 1][1], tagged_sentence[i][1]
                complex_transition_matrix[curr_pos][prev_pos][prev_prev_pos] += 1
        # Normalize
        for pos, next_pos_dict in complex_transition_matrix.items():
            complex_transition_matrix[pos] = POSHelper.normalize(next_pos_dict)

        return complex_transition_matrix

    @staticmethod
    def generate_emission_matrix(tagged_sentences):
        emission_dict = {}
        # Nested emission matrix - P(Wi | Si)
        for tagged_sentence in tagged_sentences:
            for curr_word, curr_partofspeech in tagged_sentence:
                if curr_word not in emission_dict:
                    emission_dict[curr_word] = {curr_partofspeech: 1}
                else:
                    if curr_partofspeech not in emission_dict[curr_word]:
                        emission_dict[curr_word].update({curr_partofspeech: 1})
                    else:
                        emission_dict[curr_word][curr_partofspeech] += 1

        emission_matrix = DataFrame(emission_dict).fillna(0).T

        # Normalize
        for col in emission_matrix.columns:
            emission_matrix[col] = emission_matrix[col] / emission_matrix[col].sum()

        return emission_matrix

    @staticmethod
    def generate_complex_emission_matrix(tagged_sentences):
        complex_emission_matrix = {}
        # P(Wi | Si, Si-1)
        for tagged_sentence in tagged_sentences:
            for i in range(len(tagged_sentence)):
                prev_partofspeech, curr_partofspeech, curr_word = START_TAG if i == 0 else tagged_sentence[i - 1][1], \
                                                                  tagged_sentence[i][1], tagged_sentence[i][0]
                if curr_word not in complex_emission_matrix:
                    complex_emission_matrix[curr_word] = {prev_partofspeech: {curr_partofspeech: 1.0}}
                else:
                    if prev_partofspeech not in complex_emission_matrix[curr_word]:
                        complex_emission_matrix[curr_word].update({prev_partofspeech: {curr_partofspeech: 1.0}})
                    else:
                        if curr_partofspeech not in complex_emission_matrix[curr_word][prev_partofspeech]:
                            complex_emission_matrix[curr_word][prev_partofspeech].update({curr_partofspeech: 1.0})
                        else:
                            complex_emission_matrix[curr_word][prev_partofspeech][curr_partofspeech] += 1.0

        # Normalize
        for word, nested_dict in complex_emission_matrix.items():
            complex_emission_matrix[word] = POSHelper.normalize(nested_dict)

        return complex_emission_matrix

    @staticmethod
    def normalize(d):
        for inner_dict in d.values():
            count_sum = sum(inner_dict.values())
            for k1 in inner_dict:
                inner_dict[k1] = inner_dict[k1] / count_sum
        return d

    @staticmethod
    def assign_rule_based_tag(word):
        # suffix based matching for adverbs, verbs and adjectives.
        if findall(r'[a-z]+(ly|wise|wards?)$', word):
            return 'adv'
        if findall(r'[a-z]+(able|ible|ian|ful|ous|al|ive|less|like|ish|one)$', word):
            return 'adj'
        if findall(r'[a-z]+(ing|ed|es|en)$', word):
            return 'verb'
        if findall(r'\$?\d+(.\d+)?$', word):  # Regexp for matching digits/USD
            return 'num'
        if word in punctuation:
            return '.'
        return 'noun'
