import string

import pronouncing
from Phyme import Phyme
phyme = Phyme()

from constants import *

def is_iambic(phrase):
    """
    check that we satisfy iambic meter.
    return 1 if so, otherwise 0. 
    definitely an imperfect check...
    if we end up needing to check a word that's not in the CMU dictionary, just return 0. 
    """
    meter = ''
    for word in phrase.split():
        word = word.strip().strip(string.punctuation).lower()
        try:
            phones_list = pronouncing.phones_for_word(word)
            stresses = pronouncing.stresses(phones_list[0])
            if len(stresses) == 1:
                if stresses == '1':
                    stresses = '2' # allow ambiguity for 1-syllable words with stress 1
            meter += stresses # just default to the first pronunciation if > 1 given
        except:
            return 0 # word not found
    meter = [int(x) for x in meter]
    even_stresses_full = [meter[i] for i in range(0, len(meter), 2)]
    odd_stresses_full = [meter[i] for i in range(1, len(meter), 2)]
    even_stresses = set(even_stresses_full)
    odd_stresses = set(odd_stresses_full)
    if 0 in odd_stresses:
        return 0
    if 1 in even_stresses:
        return 0
    return 1


def count_syllables(words):
    syllables = 0
    for word in words.split():
        word = word.strip().strip(string.punctuation)
        try:
            phones_list = pronouncing.phones_for_word(word)
            stresses = pronouncing.stresses(phones_list[0])
            syllables += min(MAX_SYLLABLES_PER_WORD, len(stresses))
        except:
            # if we don't know, just do a quick approximation here; it shouldn't come up too often
            syllables += min(MAX_SYLLABLES_PER_WORD, round(len(word) / 3))
    return syllables


def get_rhymes(word):
    # throws exception if word not in the rhyme dict (rare)
    rhymes = []
    rhyme_dict = phyme.get_perfect_rhymes(word)
    for length_dict in rhyme_dict.values():
        for word in length_dict:
            if '(' in word: # sometimes you have stuff like preferred(1) where they indicate a particular pronunciation
                rhymes.append(word.split('(')[0])
            else:
                rhymes.append(word)
    return sorted(list(set(rhymes)))


def get_rhyme_group(word):
    sorted_rhyme_list = get_rhymes(word)
    return ' '.join(sorted_rhyme_list)


def perfect_rhyme_end(s1, s2):
    ending_word1 = s1.split()[-1].strip(string.punctuation)
    ending_word2 = s2.split()[-1].strip(string.punctuation)
    try:
        return get_rhyme_group(ending_word1) == get_rhyme_group(ending_word2)
    except:
        return False # unknown words

if __name__=='__main__':
    result = is_iambic('Shall I compare thee to a summer day')
    result2 = count_syllables('Shall I compare thee to a summer day')
    import pdb; pdb.set_trace()