# -*- encoding: utf-8 -*-
import json
import os
import pickle
import time
from multiprocessing import Queue, Process

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from hears_patterns import HearstPatterns

"""
note
1 上位词用np匹配
2 下位词可能不是np， 所以保守用，分割，同时保留用and 和or的所有可能性
"""


class Master:
    def __init__(self, q_output, corpus_path):
        self.q_output = q_output
        self.corpus_path = corpus_path
        self.point = {}
        self.save_line = 100000
        self.save_dir = 'data'
        self.break_point_name = "master_point.json"

    def __call__(self, *args, **kwargs):
        self.load_point()
        iter_num = 0
        skip_num = self.point.get(self.corpus_path, 0)
        print("{} skip {}".format(self.corpus_path, skip_num))
        is_first_skip = True # 为了第二次迭代从头开始
        while 1:
            iter_num += 1
            c_counter = tqdm(desc="interator {}".format(iter_num))
            for sent in self.get_sentence():
                c_counter.update()
                if not sent.strip():
                    continue
                if is_first_skip and c_counter.n < skip_num:
                    continue
                is_first_skip = False
                self.q_output.put((iter_num, sent))
                if c_counter.n % self.save_line == 0:
                    self.save_point(c_counter.n)

    def get_sentence(self) -> list:
        """分句子"""
        with open(self.corpus_path, encoding='utf8') as f:
            for line in f:
                data = line.rstrip('\n').split('\n')
                for d in data:
                    if not d:
                        continue
                    for sent in sent_tokenize(d):
                        yield sent

    def save_point(self, number):
        self.point[self.corpus_path] = number
        with open(os.path.join(self.save_dir, self.break_point_name), 'w') as f:
            json.dump(self.point, f)

    def load_point(self):
        file_path = os.path.join(self.save_dir, self.break_point_name)
        if not os.path.exists(file_path):
            return
        with open(file_path) as f:
            self.point = json.load(f)


class Worker:
    def __init__(self, q_input, q_output):
        self.q_input = q_input
        self.q_output = q_output

    def __call__(self, *args, **kwargs):
        self.hp = HearstPatterns(extended=True)
        while 1:
            iter_num, sent = self.q_input.get()
            x, y = self.syntactic_extraction(sent)
            self.q_output.put((iter_num, x, y))

    def syntactic_extraction(self, sent: str) -> (list, list):
        """句子抽取x, y.  x可以无序， y必须有序，为了后面判断离match phrase最近"""
        x, y = set(), []
        hyponyms = self.hp.find_hyponyms(sent)
        for k, v in hyponyms:
            x.add(v)
            y.append(k)
        return list(x), y


class Calculate:
    def __init__(self, q_input):
        self.q_input = q_input
        self.n_super_concept = {}
        self.n_super_concept_sub_concept = {}
        self.knowledge_base_size = 1
        self.epsilon = 0.001
        self.threshold_super_concept = 1.2
        self.threshold_k = 0.02
        self.save_dir = 'data'
        self.break_point_name = 'calculate_point.pkl'
        self.output_name = 'output.txt'

    def load(self):
        save_point_path = os.path.join(self.save_dir, self.break_point_name)
        if os.path.exists(save_point_path):
            print('loading calculate point ...')
            with open(save_point_path, 'rb') as f:
                n_super_concept, n_super_concept_sub_concept, knowledge_base_size = pickle.load(f)
            return n_super_concept, n_super_concept_sub_concept, knowledge_base_size
        return {}, {}, 1

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

    def sub_concept_detection(self, super_concept, sub_concepts):
        scores = [self.p_y_x(_, super_concept) for _ in sub_concepts]
        max_score = max(scores)
        if max_score < self.threshold_k:
            return sub_concepts[:1]

        return sub_concepts[:scores.index(max_score) + 1]

    @staticmethod
    def increase_count(dictionary, key):
        """Increases count of key in dictionary"""
        dictionary[key] = dictionary.get(key, 0) + 1

    def save_file(self, iter_num=None, n_super_concept=None, n_super_concept_sub_concept=None, knowledge_base_size=1):
        """Saves probase as filename in text format"""
        if not n_super_concept:
            n_super_concept = self.n_super_concept
        if not n_super_concept_sub_concept:
            n_super_concept_sub_concept = self.n_super_concept_sub_concept

        filename = os.path.join(self.save_dir, self.output_name)
        if iter_num:
            filename += "_iter_{}".format(iter_num)
        with open(filename, 'w') as file:
            for key, value in n_super_concept_sub_concept.items():
                file.write(key[0] + '##' + key[1] + '##' + str(value) + '\n')
        with open(os.path.join(self.save_dir, self.break_point_name), 'wb') as f:
            pickle.dump((n_super_concept, n_super_concept_sub_concept, knowledge_base_size), f)

    def __call__(self, *args, **kwargs):
        n_super_concept_new, n_super_concept_sub_concept_new, knowledge_base_size_new = self.load()
        origin_iter_num = 1
        save_time = time.time()
        while 1:
            iter_num, x, y = self.q_input.get()
            if not x:
                continue

            if iter_num != origin_iter_num:
                """新一轮迭代"""
                size_old = len(self.n_super_concept_sub_concept)
                size_new = len(n_super_concept_sub_concept_new)
                if size_new == size_old:
                    break
                else:
                    self.n_super_concept_sub_concept = n_super_concept_sub_concept_new
                    self.n_super_concept = n_super_concept_new
                    self.knowledge_base_size = knowledge_base_size_new
                    n_super_concept_sub_concept_new = {}
                    n_super_concept_new = {}
                    knowledge_base_size_new = 1

            if len(x) > 1:
                most_likely_super_concept = self.super_concept_detection(x, y)
                if not most_likely_super_concept:
                    continue
            else:
                most_likely_super_concept = x[0]
            y = self.sub_concept_detection(most_likely_super_concept, y)
            for sub_concept in y:
                self.increase_count(n_super_concept_sub_concept_new,
                                    (most_likely_super_concept.chunk, sub_concept))
                self.increase_count(n_super_concept_new, most_likely_super_concept.chunk)
                knowledge_base_size_new += 1
            if time.time() - save_time > 2 * 60:
                self.save_file(origin_iter_num, n_super_concept_new, n_super_concept_sub_concept_new, knowledge_base_size_new)
                save_time = time.time()
        self.save_file(origin_iter_num, n_super_concept_new, n_super_concept_sub_concept_new, knowledge_base_size_new)
        sys.exit()


import sys

q_m2w = Queue(maxsize=3000)
q_w2c = Queue(maxsize=3000)
worker_num = 15
ps = []
ps.append(Process(target=Master(q_m2w, sys.argv[1])))
ps += [Process(target=Worker(q_m2w, q_w2c)) for _ in range(worker_num)]
ps.append(Process(target=Calculate(q_w2c)))

for p in ps:
    p.start()
for p in ps:
    p.join()
