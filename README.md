## Part-of-speech Tagging

### Problem Formulation

- The goal of this problem is to design a model that accurately predicts the part-of-speech associated with each word in
  a given sentence.
- In doing so, we make some assumptions, which are as follows:
    - All the words, and the sentences are grammatically correct.
    - The parts-of-speech are limited
      to `S = {'noun', 'num', 'pron', 'verb', 'adp', 'det', 'x', '.', 'adj', 'adv', 'prt', 'conj'}`.
- We approach this problem by solving 3 different probabilistic models, each one working a different underlying Bayesian
  Network. Each network assumes a different probabilistic relation between the set of words `W = {w1,...,wN}`
  and the set of parts-of-speech `S = {s1,...,sN}`.
- The task then, is to decode (factorize) each of the three Bayesian Networks to calculate the _Maximum a-posteriori
  estimation (MAP)_ `P(S | W)`and correctly assign a part-of-speech to a given word.
- Since the three models operate of different Bayesian Networks, we need different methods to calculate the _MAP_.

### Program Description

#### Simplified Model using Naive Bayes

- Given a set of words `W = {w1,...,wN}`, we loop through the set and calculate its maximum-likelihood using Baye's
  Theorem. `P (si | wi ) = [P(si) * P (wi | si)] / P(si) `. Since we are calculating the map, we can ignore the
  evidence (
  which is the denominator) and calculate the posterior using the equation `si* = argmax(si) P(Si = si|W).`. This is
  shown in the snippet below.

```
 for pos in parts_of_speech:
    likelihood = e_matrix.loc[word, pos]
    curr_posterior = (priors[pos] * likelihood)  # Baye's Law (Ignore the evidence for max likelihood)
    if curr_posterior > max_posterior:
        max_posterior, most_likely_pos = curr_posterior, pos
```

- To construct the logarithm of joint probability, we loop through the output of the model and
  calculate `sum[log( P(si) * P (wi | si) )]`

```
for idx, word in enumerate(sentence):
  log_joint_p += log10(priors[label[idx]] * e_matrix.loc[word, label[idx]] if (
          word in e_matrix.index and e_matrix.loc[word, label[idx]] > 0) else DEFAULT_EMISSION_P)
```

#### HMM Decoding using Viterbi

- This Bayesian Network is a Hidden Markov Model, hence we can use the Viterbi Algorithm to calculate the MAP. In doing
  so, we make use of the emission probabilities `P(Wi | Si)` and the transition probabilities `P(Si | Si-1)`. In the
  code, these are stored as simple lookup tables using a `DataFrame`, generated during the training phase.

```
self.transition_matrix, self.priors = POSHelper.generate_transition_matrix_and_priors(tagged_sentences)
self.emission_matrix = POSHelper.generate_emission_matrix(tagged_sentences)
```

- In implementing Viterbi, we move these probabilities into a Logarithmic Space, and construct a graph where the edge
  weights represent the log of probabilities. Calculating the _MAP_, then becomes a minimum-cost path finding problem
  which can be solved using Dynamic Programming.
- We maintain two additional local lookup tables , which store the optimal solutions to the sub-problems (in this case,
  the best path up to a given node, for each node). The table construction is shown in the code snippet below (
  the `viterbi_table` is initially populated with the prior probabilities.

```
viterbi_table = zeros((num_labels, num_observations))
path_table = zeros((num_labels, num_observations))
.
.
.
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
```

- The _MAP_ is then a sequence constructed by traversing the lookup tables.

```
viterbi_seq = [0 for _ in range(num_observations)]
viterbi_seq[num_observations - 1] = argmin(viterbi_table[:, -1])

for i in range(num_observations - 2, -1, -1):
    viterbi_seq[i] = int(path_table[viterbi_seq[i + 1]][i])

most_likely_sequence = [parts_of_speech[i] for i in viterbi_seq]
```

- To construct the logarithm of joint probability, we loop through the output of the model and
  calculate `log(P(s0)) + sum [ log( P(si-1 | si) * P (wi | si) ) ]`

#### Markov Chain Monte Carlo (MCMC) using Gibbs' Sampling

- This Bayesian Networks adds more complexity by factoring in more conditional dependencies between the parts-of-speech
  and the words.
- In particular, a part-of-speech `si` depends on `si-1 and si-2`. Moreover, the emission probability of a word also
  considers the previous part-of-speech `si-1`.
- This gives us the following factorization `P(si-1 | si ) * P(si | si-1,si-2) * P(wi | si) * P (wi | si-1, si)`. To
  store these additional probabilities, we construct two more lookup structures `complex_transition_matrix`
  and `complex_emission_matrix`, which are 3-dimensional dictionaries.

```
.
.
.
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
.
.
.                    
```

```
.
.
.
complex_transition_matrix = {p: {p: defaultdict(int) for p in POS[1:]} for p in POS}
for tagged_sentence in tagged_sentences:
    for i in range(1, len(tagged_sentence)):
        prev_prev_pos, prev_pos, curr_pos = START_TAG if i == 1 else tagged_sentence[i - 2][1], \
                                            tagged_sentence[i - 1][1], tagged_sentence[i][1]
        complex_transition_matrix[curr_pos][prev_pos][prev_prev_pos] += 1
.
.
.
```

- To implement Gibbs sampling, we start with an initial sample of all nouns, and generate new samples using the
  factorizations and the tables mentioned above. This is done for each word, by assigning different parts-of-speech to
  each it, and calculating the max-marginal.

```
most_likely_sequence = ['noun'] * len(sentence)
for _ in range(120):
    most_likely_sequence = self.generate_sample(sentence, most_likely_sequence)
return most_likely_sequence
.
.
.
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

```

- To construct the logarithm of joint probability, we loop through the output of the model and
  calculate `sum [ log( P(si-1 | si ) * P(si | si-1,si-2) * P(wi | si) * P (wi | si-1, si) ) ]`

#### Results

```
==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
         1. Simple:       94.21%               49.00%
            2. HMM:       96.10%               61.05%
        3. Complex:       89.04%               29.15%
----

```

### Problems Faced / Design Decisions / Comments

- To handle missing evidence, i.e. calculating the emission probability of a word that has not been seen before, we use
  grammatical rules to determine the best possible match for the given word. This is implemented using regular
  expressions.
- We also assume a probability, `DEFAULT_EMISSION_P = 0.0000000000001` for such words.

```
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
```
