import re
from chunk import Chunk

import spacy


class HearstPatterns(object):
    def __init__(self, extended=False):
        self.__adj_stopwords = ['able', 'available', 'brief', 'certain', 'different', 'due', 'enough', 'especially',
                                'few', 'fifth', 'former', 'his', 'howbeit', 'immediate', 'important', 'inc', 'its',
                                'last', 'latter', 'least', 'less', 'likely', 'little', 'many', 'ml', 'more', 'most',
                                'much', 'my', 'necessary', 'new', 'next', 'non', 'old', 'other', 'our', 'ours', 'own',
                                'particular', 'past', 'possible', 'present', 'proud', 'recent', 'same', 'several',
                                'significant', 'similar', 'such', 'sup', 'sure']

        # now define the Hearst patterns
        # format is <hearst-pattern>, <general-term>
        # so, what this means is that if you apply the first pattern, the firsr Noun Phrase (NP)
        # is the general one, and the rest are specific NPs
        self.__hearst_patterns = [
            ('(NP_\\w+ (, )?such as (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
            ('(such NP_\\w+ (, )?as (NP_\\w+ ?(, )?(and |or )?)+)', 'first'),
            ('((NP_\\w+ ?(, )?)+(and |or )?other NP_\\w+)', 'last'),
            ('(NP_\\w+ (, )?include (NP_\\w+ ?(, )?(and |or )?)+)', 'first'),
            ('(NP_\\w+ (, )?especially (NP_\\w+ ?(, )?(and |or )?)+)', 'first'),
        ]

        if extended:
            self.__hearst_patterns.extend([
                ('((NP_\\w+ ?(, )?)+(and |or )?any other NP_\\w+)', 'last'),
                ('((NP_\\w+ ?(, )?)+(and |or )?some other NP_\\w+)', 'last'),
                ('((NP_\\w+ ?(, )?)+(and |or )?be a NP_\\w+)', 'last'),
                ('(NP_\\w+ (, )?like (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('such (NP_\\w+ (, )?as (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_\\w+ ?(, )?)+(and |or )?like other NP_\\w+)', 'last'),
                ('((NP_\\w+ ?(, )?)+(and |or )?one of the NP_\\w+)', 'last'),
                ('((NP_\\w+ ?(, )?)+(and |or )?one of these NP_\\w+)', 'last'),
                ('((NP_\\w+ ?(, )?)+(and |or )?one of those NP_\\w+)', 'last'),
                ('example of (NP_\\w+ (, )?be (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_\\w+ ?(, )?)+(and |or )?be example of NP_\\w+)', 'last'),
                ('(NP_\\w+ (, )?for example (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_\\w+ ?(, )?)+(and |or )?wich be call NP_\\w+)', 'last'),
                ('((NP_\\w+ ?(, )?)+(and |or )?which be name NP_\\w+)', 'last'),
                ('(NP_\\w+ (, )?mainly (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?mostly (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?notably (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?particularly (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?principally (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?in particular (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?except (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?other than (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?e.g. (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?i.e. (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_\\w+ ?(, )?)+(and |or )?a kind of NP_\\w+)', 'last'),
                ('((NP_\\w+ ?(, )?)+(and |or )?kind of NP_\\w+)', 'last'),
                ('((NP_\\w+ ?(, )?)+(and |or )?form of NP_\\w+)', 'last'),
                ('((NP_\\w+ ?(, )?)+(and |or )?which look like NP_\\w+)', 'last'),
                ('((NP_\\w+ ?(, )?)+(and |or )?which sound like NP_\\w+)', 'last'),
                ('(NP_\\w+ (, )?which be similar to (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?example of this be (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?type (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_\\w+ ?(, )?)+(and |or )? NP_\\w+ type)', 'last'),
                ('(NP_\\w+ (, )?whether (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(compare (NP_\\w+ ?(, )?)+(and |or )?with NP_\\w+)', 'last'),
                ('(NP_\\w+ (, )?compare to (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('(NP_\\w+ (, )?among -PRON- (NP_\\w+ ? (, )?(and |or )?)+)', 'first'),
                ('((NP_\\w+ ?(, )?)+(and |or )?as NP_\\w+)', 'last'),
                ('(NP_\\w+ (, )? (NP_\\w+ ? (, )?(and |or )?)+ for instance)', 'first'),
                ('((NP_\\w+ ?(, )?)+(and |or )?sort of NP_\\w+)', 'last')
            ])

        self.__hearst_patterns = [(re.compile(k), v) for k, v in self.__hearst_patterns]
        self.__spacy_nlp = spacy.load('en')

    def chunk(self, rawtext):
        doc = self.__spacy_nlp(rawtext)
        chunks = []
        chunks_index = {}
        for sentence in doc.sents:
            sentence_text = sentence.lemma_
            for chunk in sentence.noun_chunks:
                chunk_arr = []
                for token in chunk:
                    # Ignore Punctuation and stopword adjectives (generally quantifiers of plurals)
                    if token.is_punct or token.lemma_ in self.__adj_stopwords:
                        continue
                    chunk_arr.append(token.lemma_)
                chunk_lemma = " ".join(chunk_arr)
                replacement_value = "NP_" + "_".join(chunk_arr)
                chunks_index[replacement_value] = Chunk(chunk=chunk_lemma, chunk_root=chunk.root.lemma_)
                sentence_text = sentence_text.replace(chunk_lemma, replacement_value)
            chunks.append(sentence_text)
        return chunks, chunks_index

    """
        This is the main entry point for this code.
        It takes as input the rawtext to process and returns a list of tuples (specific-term, general-term)
        where each tuple represents a hypernym pair.

    """

    def find_hyponyms(self, rawtext):

        hyponyms = []
        np_tagged_sentences, chunks_index = self.chunk(rawtext)
        for sentence in np_tagged_sentences:
            # two or more NPs next to each other should be merged into a single NP, it's a chunk error

            for (hearst_pattern, parser) in self.__hearst_patterns:
                matches = hearst_pattern.search(sentence)
                if matches:
                    match_str = matches.group(0)

                    nps = [a for a in match_str.split() if a.startswith("NP_")]

                    if parser == "first":
                        general = nps[0]
                        specifics = nps[1:]
                    else:
                        general = nps[-1]
                        specifics = nps[:-1]

                    for i in range(len(specifics)):
                        try:
                            hyponyms.append((self.clean_hyponym_term(specifics[i]), chunks_index[general]))
                        except:
                            pass

        return hyponyms

    def clean_hyponym_term(self, term):
        # good point to do the stemming or lemmatization
        return term.replace("NP_", "").replace("_", " ")
