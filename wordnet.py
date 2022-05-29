import nltk
import torch
import geoopt
import pathlib
import numpy as np
from tqdm import tqdm
from itertools import count
from sklearn import metrics
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.corpus import wordnet as wn


def generate_dataset(output_dir, with_mammal=False):
    output_path = pathlib.Path(output_dir) / 'noun_closure.tsv'

    # make sure each edge is included only once
    edges = set()
    for synset in wn.all_synsets(pos='n'):
        # write the transitive closure of all hypernyms of a synset to file
        for hyper in synset.closure(lambda s: s.hypernyms()):
            edges.add((synset.name(), hyper.name()))

        # also write transitive closure for all instances of a synset
        for instance in synset.instance_hyponyms():
            for hyper in instance.closure(lambda s: s.instance_hypernyms()):
                edges.add((instance.name(), hyper.name()))
                for h in hyper.closure(lambda s: s.hypernyms()):
                    edges.add((instance.name(), h.name()))

    with output_path.open('w') as fout:
        for i, j in edges:
            fout.write('{}\t{}\n'.format(i, j))

    if with_mammal:
        import subprocess
        mammaltxt_path = pathlib.Path(output_dir).resolve() / 'mammals.txt'
        mammaltxt = mammaltxt_path.open('w')
        mammal = (pathlib.Path(output_dir) / 'mammal_closure.tsv').open('w')
        commands_first = [
            ['cat', '{}'.format(output_path)],
            ['grep', '-e', r'\smammal.n.01'],
            ['cut', '-f1'],
            ['sed', r's/\(.*\)/\^\1/g']
        ]
        commands_second = [
            ['cat', '{}'.format(output_path)],
            ['grep', '-f', '{}'.format(mammaltxt_path)],
            ['grep', '-v', '-f', '{}'.format(
                pathlib.Path(__file__).resolve().parent / 'mammals_filter.txt'
            )]
        ]
        for writer, commands in zip([mammaltxt, mammal],
                                    [commands_first, commands_second]):
            for i, c in enumerate(commands):
                if i == 0:
                    p = subprocess.Popen(c, stdout=subprocess.PIPE)
                elif i == len(commands) - 1:
                    p = subprocess.Popen(c, stdin=p.stdout, stdout=writer)
                else:
                    p = subprocess.Popen(
                        c, stdin=p.stdout, stdout=subprocess.PIPE)
                # prev_p = p
            p.communicate()
        mammaltxt.close()
        mammal.close()


def parse_seperator(line, length, sep='\t'):
    d = line.strip().split(sep)
    if len(d) == length:
        w = 1
    elif len(d) == length + 1:
        w = int(d[-1])
        d = d[:-1]
    else:
        raise RuntimeError('Malformed input ({})'.format(line.strip()))
    return tuple(d) + (w,)


def parse_tsv(line, length=2):
    return parse_seperator(line, length, '\t')


def iter_line(file_name, parse_function, length=2, comment='#'):
    with open(file_name, 'r') as fin:
        for line in fin:
            if line[0] == comment:
                continue
            tpl = parse_function(line, length=length)
            if tpl is not None:
                yield tpl


def intmap_to_list(d):
    arr = [None for _ in range(len(d))]
    for v, i in d.items():
        arr[i] = v
    assert not any(x is None for x in arr)
    return arr


def slurp(file_name, parse_function=parse_tsv, symmetrize=False):
    ecount = count()
    enames = defaultdict(ecount.__next__)

    subs = []
    for i, j, w in iter_line(file_name, parse_function, length=2):
        if i == j:
            continue
        subs.append((enames[i], enames[j], w))
        if symmetrize:
            subs.append((enames[j], enames[i], w))
    idx = np.array(subs, dtype=np.int64)

    # freeze defaultdicts after training data and convert to arrays
    objects = intmap_to_list(dict(enames))
    print('slurp: file_name={}, objects={}, edges={}'.format(
        file_name, len(objects), len(idx)))
    return idx, objects


def create_adjacency(indices):
    adjacency = defaultdict(set)
    for i in range(len(indices)):
        s, o = indices[i]
        adjacency[s].add(o)

    return adjacency


class WordNet(Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        self.path = args.path
        file_name = pathlib.Path(self.path).expanduser() / 'mammal_closure.tsv'
        relations, words = slurp(file_name.as_posix(), symmetrize=False)
    
        self.relations = relations[:, :2]
        self.words = words
        self.n_relations = len(self.relations)
        self.n_words = len(self.words)
        
        self.graph = create_adjacency(self.relations)
        for i in range(self.n_words):
            self.graph[i] = set(range(self.n_words)) - self.graph[i]

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx):
        anchor = self.relations[idx, 0]
        negatives = np.random.choice(
            list(self.graph[anchor]),
            10,
            replace=False
        )
        return np.r_[
            self.relations[idx],
            negatives
            # np.random.randint(self.n_words, size=50)
        ]


def calculate_energy(model, x, batch_size):
    x = torch.tensor(x).cuda()
    kl_target = torch.zeros(x.size(0)).cuda()
    nb_batch = np.ceil(x.size(0) / batch_size).astype(int)

    manifold = geoopt.manifolds.Lorentz()
    for i in range(nb_batch):
        idx_start = i * batch_size
        idx_end = (i + 1) * batch_size
        data = x[idx_start:idx_end]

        embeds = model(data)
        dist = manifold.dist(embeds[:, 0], embeds[:, 1])
        kl_target[idx_start:idx_end] = dist

    return kl_target


def calculate_metrics(dataset, model):
    ranks = []
    ap_scores = []

    adjacency = create_adjacency(dataset.relations)

    iterator = tqdm(adjacency.items())
    batch_size = dataset.n_words // 100
    for i, (source, targets) in enumerate(iterator):
        # if approx and i % 100 != 0:
        if i % 1000 != 0:
            continue
        input_ = np.c_[
            source * np.ones(dataset.n_words).astype(np.int64),
            np.arange(dataset.n_words)
        ]
        _energies = calculate_energy(
            model,
            input_, 
            batch_size,
        ).detach().cpu().numpy()
        
        _energies[source] = 1e+12
        _labels = np.zeros(dataset.n_words)
        _energies_masked = _energies.copy()
        _ranks = []
        for o in targets:
            _energies_masked[o] = np.Inf
            _labels[o] = 1
        ap_scores.append(metrics.average_precision_score(_labels, -_energies))
        for o in targets:
            ene = _energies_masked.copy()
            ene[o] = _energies[o]
            r = np.argsort(ene)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks

    return np.mean(ranks), np.mean(ap_scores)

