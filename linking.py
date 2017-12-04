import spacy

nlp = spacy.load('en')


def print_len(doc):
    print("#tokens:", len(doc))
    return doc


nlp.add_pipe(print_len, name='print_len')


class CountNERType():

    def __init__(self, ner_type):
        self.ner_type = ner_type
        self.name = 'count_{}'.format(ner_type)

    def __call__(self, doc):
        cnt = len(list(filter(lambda x: x.label_ == self.ner_type, doc.ents)))
        print("#{}: {}".format(self.ner_type, cnt))
        return doc


countPERSON = CountNERType('PERSON')
countGPE = CountNERType('GPE')
countORG = CountNERType('ORG')
nlp.add_pipe(countPERSON)
nlp.add_pipe(countGPE)
nlp.add_pipe(countORG)

print(nlp.pipe_names)
inp = 'Lukas Galke is in Austin, TX and experiments with spacy'
print('INPUT:', inp)
doc = nlp(inp)
# print("\nEntities", '=' * 8, sep='\n')
# print(*list((ent, ent.label_) for ent in doc.ents), sep='\n')
