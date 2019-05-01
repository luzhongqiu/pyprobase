# -*- encoding: utf-8 -*-

import spacy
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from hears_patterns import HearstPatterns

# a contain b, 1表示正序，0表示反序
# rules = [
#     (r'(?P<A>.*) such as (?P<B>.*)', 1),
#     (r'(.*) such (?P<A>.*?) as (?P<B>.*)', 1),
#     (r'(?P<A>.*) including (?P<B>.*)', 1),
#     (r'(?P<B>.*) (and|or) other (?P<A>.*)', 0),
#     (r'(?P<A>.*) especially (?P<B>.*)', 1)
# ]

rules = [
    (r'NP such as (NP,|NP ,)* (or|and) NP', 1)
]

"""
note
1 上位词用np匹配
2 下位词可能不是np， 所以保守用，分割，同时保留用and 和or的所有可能性
"""


class Probase:
    def __init__(self, corpus_path):
        self.nlp = spacy.load('en')
        self.corpus_path = corpus_path
        self.n_super_concept = {}
        self.n_super_concept_sub_concept = {}
        self.knowledge_base_size = 1
        self.epsilon = 0.001
        self.threshold_super_concept = 1.1
        self.threshold_k = 0.01
        self.hp = HearstPatterns(extended=True)

    def run(self):
        """主程序"""
        iter_num = 0
        while 1:
            iter_num += 1
            print()
            print("Interator {}".format(iter_num))
            n_super_concept_sub_concept_new = {}
            n_super_concept_new = {}
            knowledge_base_size_new = 1
            for sent in tqdm(self.get_sentence()):
                x, y = self.syntactic_extraction(sent)
                print("x: ", x)
                print("y: ", y)
                if not x:
                    continue
                if len(x) > 1:
                    most_likely_super_concept = self.super_concept_detection(x, y)
                    if not most_likely_super_concept:
                        continue
                else:
                    most_likely_super_concept = x[0]

                y = self.sub_concept_detection(most_likely_super_concept, y)
                print("most_likely_super_concept: ", most_likely_super_concept)
                print("detect y: ", y)
                for sub_concept in y:
                    self.increase_count(n_super_concept_sub_concept_new,
                                        (most_likely_super_concept.chunk, sub_concept.chunk))
                    self.increase_count(n_super_concept_new, most_likely_super_concept.chunk)
                    knowledge_base_size_new += 1

            size_old = len(self.n_super_concept_sub_concept)
            size_new = len(n_super_concept_sub_concept_new)
            print("old size: ", size_old)
            print("new size: ", size_new)
            if size_new == size_old:
                break
            else:
                self.n_super_concept_sub_concept = n_super_concept_sub_concept_new
                self.n_super_concept = n_super_concept_new
                self.knowledge_base_size = knowledge_base_size_new

            if iter_num % 10 == 0:
                self.save_file('data/output.txt')

        self.save_file('data/output.txt')

    def save_file(self, filename):
        """Saves probase as filename in text format"""
        with open(filename, 'w') as file:
            for key, value in self.n_super_concept_sub_concept.items():
                file.write(key[0] + '\t' + key[1] + '\t' + str(value) + '\n')

    @staticmethod
    def increase_count(dictionary, key):
        """Increases count of key in dictionary"""
        if key in dictionary:
            dictionary[key] += 1
        else:
            dictionary[key] = 1

    def p_x(self, super_concept):
        """计算P(x)"""
        probability = self.n_super_concept.get(
            super_concept.chunk, 0) / self.knowledge_base_size
        if super_concept.chunk != super_concept.chunk_root:
            probability_root = self.n_super_concept.get(
                super_concept.chunk_root, 0) / self.knowledge_base_size
            probability += probability_root

        if probability == 0:
            return self.epsilon
        else:
            return probability

    def p_y_x(self, sub_concept, super_concept):
        """计算P(y|x)中所有y的乘积"""
        probability = self.n_super_concept_sub_concept.get(
            (super_concept.chunk, sub_concept), 0) / self.n_super_concept.get(super_concept.chunk, 1)
        if super_concept.chunk != super_concept.chunk_root:
            probability_root = self.n_super_concept_sub_concept.get(
                (super_concept.chunk_root, sub_concept), 0) / self.n_super_concept.get(super_concept.chunk_root, 1)
            probability += probability_root

        if probability == 0:
            return self.epsilon
        else:
            return probability

    def sub_concept_detection(self, super_concept, sub_concepts):
        scores = [self.p_y_x(_, super_concept) for _ in sub_concepts]
        max_score = max(scores)
        if max_score < self.threshold_k:
            return sub_concepts[:1]

        return sub_concepts[:scores.index(max_score) + 1]

    def super_concept_detection(self, super_concepts: list, sub_concepts: list):
        likelihoods = {}
        for super_concept in super_concepts:
            probability_super_concept = self.p_x(super_concept)
            likelihood = probability_super_concept
            for sub_concept in sub_concepts:
                probability_y_x = self.p_y_x(sub_concept, super_concept)
                likelihood *= probability_y_x
            likelihoods[super_concept] = likelihood

        sorted_likelihoods = sorted(
            likelihoods.items(), key=lambda x: x[1], reverse=True)

        """只有一个"""
        if len(sorted_likelihoods) == 1:
            return super_concepts[0]
        """如果第二个likehoods是0，取第一个"""
        if sorted_likelihoods[1][1] == 0:
            return sorted_likelihoods[0][0]
        """比较2个最大的"""
        ratio = sorted_likelihoods[0][1] / sorted_likelihoods[1][1]
        if ratio > self.threshold_super_concept:
            return sorted_likelihoods[0][0]
        else:
            return None
        pass

    def get_sentence(self) -> list:
        """分句子"""
        with open(self.corpus_path, encoding='utf8') as f:
            for line in f:
                for sent in sent_tokenize(line.rstrip('\n')):
                    yield sent

    def syntactic_extraction(self, sent: str) -> (list, list):
        """句子抽取x, y.  x可以无序， y必须有序，为了后面判断离match phrase最近"""
        x, y = set(), []

        hyponyms = self.hp.find_hyponyms(sent)
        for k, v in hyponyms:
            x.add(v)
            y.append(k)
        return list(x), y


probase = Probase('data/input.txt')
probase.run()
