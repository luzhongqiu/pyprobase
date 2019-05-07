# -*- encoding: utf-8 -*-
import re
import time

import enchant
import spacy
from nltk import WordNetLemmatizer

from chunk import Chunk


class Extractor:
    dic = enchant.Dict('en_US')

    def __init__(self):
        self.noun_chunk_pattern = r'<[A-Z]*>(<HYPH><[A-Z]*>)+|(<JJ>|<VBG>|<VBN>)*(<NN[A-Z]*>)+'
        self.lemmatizer = WordNetLemmatizer()
        self.__spacy_nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])

    def chunk(self, one_doc):
        text = one_doc.replace("\n", " ")
        text = text.strip()
        doc = self.__spacy_nlp(text)
        terms_index = self.tag_regex_matches(doc)

        noun_chunk = []
        content = []
        current_index = 0
        for start, end in terms_index:
            _noun_list = str(doc[start: end]).split(" ")
            _noun_list = [self.lemmatizer.lemmatize(str(word)) for word in _noun_list]
            _noun = "_".join(_noun_list)
            # 再清洗
            if len(_noun_list) > 1:
                _noun = self.clean_again(_noun)  # fixme
                if _noun is None:
                    continue
            _noun = "NP_" + _noun
            noun_chunk.append(_noun)
            # word_rest 不做 lemmatize
            word_rest = [_ for _ in str(doc[current_index: start]).split(" ")]
            content += word_rest + [_noun]
            current_index = end

        word_rest = [self.lemmatizer.lemmatize(_) for _ in str(doc[current_index:]).split(" ")]
        content += word_rest
        content = [_ for _ in content if _.strip()]

        # 包装输出
        chunk_index = {k: Chunk(chunk=k.strip('NP_'), chunk_root=k.strip('NP_').split('_')[-1]) for k in noun_chunk}
        sentence = " ".join(content)
        return [sentence], chunk_index

    def tag_regex_matches(self, doc, pattern=None, debug=False):
        pattern = pattern or self.noun_chunk_pattern
        pattern = re.sub(r'\s', '', pattern)
        pattern = re.sub(r'<([^>]+)>', r'( \1)', pattern)
        tags = ' ' + ' '.join(tok.tag_ for tok in doc)
        for m in re.finditer(pattern, tags):
            yield tags[0:m.start()].count(' '), tags[0:m.end()].count(' ')

    @classmethod
    def clean_again(cls, phrase):
        phrases = [phrase]
        phrase_upper, phrase_true, phrase_false, phrase_digit = cls.phrase_classiffication(phrases)
        phrase_diagonal, phrase_point, phrase_low_length, phrase_correct, phrase_number, phrase_slash = cls.phrase_deeper(
            phrase_true)
        if phrase_correct:
            return phrase_correct[0]
        return

    @classmethod
    def phrase_classiffication(cls, phrases):
        phrase_upper, phrase_true, phrase_false, phrase_digit = [], [], [], []
        for word in phrases:
            try:
                word_list = re.split(r'-|_|/', word)
                if all(cls.is_float(item) or item.isdigit() for item in word_list if item.strip()):
                    phrase_digit.append(word)
                elif all(cls.dic.check(item) for item in word_list if item.strip()):
                    phrase_true.append(word)
                elif all(cls.dic.check(item) or item.isupper() for item in word_list if item.strip()):
                    phrase_upper.append(word)
                else:
                    phrase_false.append(word)
            except:
                pass
        return phrase_upper, phrase_true, phrase_false, phrase_digit

    @staticmethod
    def is_float(s):
        return sum([n.isdigit() for n in s.strip().split('.')]) >= 2

    @classmethod
    def phrase_deeper(cls, phrase_true):
        phrase_diagonal, phrase_point, phrase_low_length, phrase_correct, phrase_number, phrase_slash = [], [], [], [], [], []
        for word in phrase_true:
            word_list = re.split(r'_|-|/', word)
            if word.startswith('-') or word.endswith('-'):
                phrase_diagonal.append(word)
            elif '.' in word and '/' not in word:
                phrase_point.append(word)
            elif '/' in word:
                phrase_slash.append(word)
            elif any(len(item) <= 2 for item in word_list):
                phrase_low_length.append(word)
            elif any(number in word for number in '1234567890'):
                phrase_number.append(word)
            else:
                phrase_correct.append(word)
        return phrase_diagonal, phrase_point, phrase_low_length, phrase_correct, phrase_number, phrase_slash


if __name__ == '__main__':
    ex = Extractor()
    ex.chunk(
        "His announcement took observers by surprise and sent markets into a tailspin, but Mr Lighthizer and Mr Mnuchin’s comments showed the impending tariffs as less an impulsive move by the President, and more fuelled by frustrations arising from the deeper disagreements between Beijing and Washington. They also lessened doubts that the President had been bluffing.")
