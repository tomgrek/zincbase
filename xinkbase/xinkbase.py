from collections import deque, defaultdict
import copy
import csv
import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from torch.utils.data import DataLoader
import torch

from logic.Term import Term
from logic.Rule import Rule
from logic.Goal import Goal
from logic.common import unify, process
from nn.dataloader import TrainDataset, BidirectionalOneShotIterator
from nn.rotate import KGEModel
from utils.string_utils import strip_all_whitespace

class KB(object):
    """Knowledge Base Class

    >>> kb = KB()
    >>> kb.__class__
    <class 'xinkbase.KB'>
    """
    def __init__(self):
        self.G = nx.DiGraph()
        self.rules = []
        self._entity2id = {}
        self._relation2id = {}
        self._encoded_triples = []
        self._kg_model = None
        self._knn = None
        self._knn_index = []
        self._cuda = False
        self.classifiers = {}

    def seed(self, seed):
        """Seed the RNGs for PyTorch, NumPy, and Python itself.
        
        :param int seed: random seed

        :Example:

        >>> KB().seed(555)
        """
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def _valid_neighbors(self, node, reverse=False):
        if reverse:
            graph = self.G.reverse()
        else:
            graph = self.G
        neighbors = graph[node]
        return [x for x in neighbors.items()]

    def bfs(self, start_node, target_node, max_depth=10, reverse=False):
        """Find a path from start_node to target_node"""
        stack = [(start_node, 0, [])]
        answers = []
        while stack:
            node, depth, path = stack.pop(0)
            if depth >= max_depth:
                return answers
            for n, pred in self._valid_neighbors(node, reverse=reverse):
                if n == target_node:
                    yield path + [(pred['pred'], n)]
                else:
                    stack.append((n, depth+1, path + [(pred['pred'], n)]))
        return answers

    def add_node_to_trained_kg(self, sub, pred, ob):
        if (sub not in self._entity2id and ob not in self._entity2id) or (pred not in self._relation2id):
            raise Exception('Must have at least a known predicate and one of subject/object in the graph already.')
        known_sub = False
        if sub in self._entity2id:
            known_sub = True
        embeddings_copy = self._kg_model.entity_embedding.clone().detach().requires_grad_(True)
        new_embed = torch.zeros((1, embeddings_copy.shape[1]), requires_grad=False)
        if known_sub:
            nodes = self.query('{}({}, X)'.format(pred, sub))
            self._entity2id[ob] = len(self._entity2id)
        else:
            nodes = self.query('{}(X, {})'.format(pred, ob))
            self._entity2id[sub] = len(self._entity2id)
        for node in nodes:
            new_embed += self.get_embedding(node['X'])
        new_embed /= len(nodes)
        # TODO: Relations have embeddings also; add the relation embedding to new_embed each
        # time and average it.
        new_embed = new_embed.clone().detach().requires_grad_(True)
        self.store('{}({}, {})'.format(pred, sub, ob))
        self._kg_model.entity_embedding = torch.nn.Parameter(torch.cat((embeddings_copy, new_embed)))

    def create_multi_classifier(self, pred):
        all_examples = self.query('{}(X, Y)'.format(pred))
        Xs = []
        Ys = []
        indexes = list(set([x['Y'] for x in all_examples]))
        ratios = defaultdict(int)
        for example in all_examples:
            Xs.append(self.get_embedding(example['X']))
            Ys.append(indexes.index(example['Y']))
            ratios[indexes.index(example['Y'])] += 1
        Xs = np.reshape(np.stack(Xs), (-1, self.get_embedding(all_examples[0]['X']).shape[1]))
        Ys = np.stack(Ys)
        num_in_biggest_class = max(v for (k, v) in ratios.items())
        for ratio in ratios:
            ratios[ratio] = num_in_biggest_class / ratios[ratio]
        clf = SVC(gamma='auto', kernel='linear', class_weight=ratios)
        clf.fit(Xs, Ys)
        self.classifiers[pred] = (clf, indexes)
        return clf

    def multi_classify(self, subject, pred):
        clf, indexes = self.classifiers[pred]
        return indexes[int(clf.predict(np.reshape(self.get_embedding(subject), (1, -1))))]

    def create_binary_classifier(self, pred, object):
        all_examples = self.query('{}(X, Y)'.format(pred))
        pos_examples = [self.get_embedding(x['X']) for x in all_examples if x['Y'] == object]
        neg_examples = [self.get_embedding(x['X']) for x in all_examples if x['Y'] != object]
        Xs = np.reshape(np.stack(pos_examples + neg_examples), (-1, pos_examples[0].shape[1]))
        Ys = np.stack([2 for x in pos_examples] + [1 for x in neg_examples])
        ratio = int(len(neg_examples) / len(pos_examples))
        clf = SVC(gamma='auto', kernel='linear', class_weight={2:min(ratio, 15)})
        clf.fit(Xs, Ys)
        self.classifiers[(pred, object)] = clf
        return clf

    def binary_classify(self, subject, pred, object):
        clf = self.classifiers[(pred, object)]
        X = self.get_embedding(subject)
        pred = int(clf.predict(X))
        return pred == 2

    def build_kg_model(self, cuda=False, embedding_size=256, gamma=2, model_name='RotatE', node_attributes=[]):
        """Build the dictionaries and KGE model
        :param list node_attributes: List of node attributes to include in the model. \
        If node doesn't possess the attribute, will be treated as zero. So far attributes \
        must be floats.
        """
        triples = self.to_triples(data=True)
        for i, triple in enumerate(triples):
            if triple[0] not in self._entity2id:
                self._entity2id[triple[0]] = len(self._entity2id)
        for i, triple in enumerate(triples):
            if triple[1] not in self._relation2id:
                self._relation2id[triple[1]] = len(self._relation2id)
        curlen = len(self._entity2id)
        j = 0
        for i, triple in enumerate(triples):
            if triple[2] not in self._entity2id:
                self._entity2id[triple[2]] = curlen + j
                j += 1
        self._encoded_triples = []
        for triple in triples:
            # TODO: only encoding a single attribute here and it must be a float; provide for optional number
            # of attributes and for a dictionary encoding of them (for categoricals)
            # TODO: check this still works if user doesn't want any attributes, only graph structure.
            attrs = []
            for attribute in node_attributes:
                # currently to_triples returns (sub, pred, ob, sub_attrs)
                # TODO: extend it to pred_attrs and ob_attrs
                attr = float(triple[3].get(attribute, 0.0))
                attrs.append(attr)
            self._encoded_triples.append((self._entity2id[triple[0]], self._relation2id[triple[1]], self._entity2id[triple[2]], attrs))
        dee = False; dre = False
        if model_name == 'ComplEx':
            dee = True
            dre = True
        if model_name == 'RotatE':
            dee = True
            dre = False
        if cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        self._kg_model = KGEModel(model_name=model_name,
                             nentity=len(self._entity2id),
                             nrelation=len(self._relation2id),
                             hidden_dim=embedding_size,
                             gamma=gamma,
                             double_entity_embedding=dee,
                             double_relation_embedding=dre,
                             node_attributes=node_attributes,
                             device=device)
        if cuda:
            self._cuda = True
            self._kg_model = self._kg_model.cuda()

    def train_kg_model(self, steps=1000, batch_size=512, lr=0.001, reencode_triples=False):
        """Train a KG model on the KB.

        :param int steps: Number of training steps
        :param int batch_size: Batch size for training
        :param float lr: Initial learning rate for Adam optimizer
        :param bool reencode_triples: If a node has been added since last training, set this to True
        """
        if reencode_triples:
            # TODO: this is not encoding attributes as well, yet.
            triples = self.to_triples(data=True)
            self._encoded_triples = []
            for triple in triples:
                self._encoded_triples.append((self._entity2id[triple[0]], self._relation2id[triple[1]], self._entity2id[triple[2]]))

        nentity = len(self._entity2id)
        nrelation = len(self._relation2id)
        # 4 negative examples per positive seems to work well.
        train_dataloader_head = DataLoader(
                    TrainDataset(self._encoded_triples, nrelation, 4, 'head-batch'),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    collate_fn=TrainDataset.collate_fn)
        train_dataloader_tail = DataLoader(
                    TrainDataset(self._encoded_triples, nrelation, 4, 'tail-batch'),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    collate_fn=TrainDataset.collate_fn)
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._kg_model.parameters()), lr=lr)
        for step in range(0, steps):
            log = self._kg_model.train_step(self._kg_model, optimizer, train_iterator, {'cuda': self._cuda})
            if step % 100 == 0:
                print(log)

    def estimate_triple_prob(self, sub, pred, ob):
        if not self._kg_model:
            raise Exception('Must build and train the model first')
        tensor = torch.tensor([[self._entity2id[sub], self._relation2id[pred], self._entity2id[ob]]])
        if self._cuda:
            tensor = tensor.cuda()
        logit, _ = self._kg_model(tensor, attributes=False)
        return round(expit(float(logit)), 4)

    def get_embedding(self, entity):
        index = torch.LongTensor([self._entity2id[entity]])
        if self._cuda:
            index = index.cuda()
        return torch.index_select(self._kg_model.entity_embedding, dim=0, index=index).detach()

    def fit_knn(self, entities):
        self._knn_index = []
        encoded_entities = []
        for e in entities:
            encoded_entities.append(self._entity2id[e])
            self._knn_index.append(e)
        index = torch.LongTensor(encoded_entities)
        if self._cuda:
            index = index.cuda()
        embeddings = torch.index_select(self._kg_model.entity_embedding, dim=0, index=index).detach().cpu()
        self._knn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(embeddings)

    def get_nearest_neighbors(self, entity, k=1):
        embedding = self.get_embedding(entity)
        embedding = embedding.cpu() # no cuda for sklearn
        distances, indices = self._knn.kneighbors(embedding, n_neighbors=k)
        borgs = []
        distances = distances[0]
        indices = indices[0]
        for i in range(len(distances)):
            borgs.append({'distance': round(distances[i], 4), 'entity': self._knn_index[int(indices[i])]})
        return borgs

    def get_most_likely(self, sub, pred, ob, k=1):
        orig_sub = sub
        orig_ob = ob
        if sub == '?':
            sub = 'X'
            ob = 'Y'
        else:
            ob = 'X'
            sub = 'Y'
        # note, this doesn't really work for queries like ? lives_in seattle
        # because the candidates from the line below are only people that
        # already have a lives_in relation TODO.
        candidates = self.query('{}({}, {})'.format(pred, sub, ob))
        possibles = []
        candidates = list(set([x['X'] for x in candidates]))
        reverse_lookup = {}
        for cand in candidates:
            reverse_lookup[self._entity2id[cand]] = cand
            if orig_sub == '?':
                possibles.append([self._entity2id[cand], self._relation2id[pred], self._entity2id[orig_ob]])
            else:
                possibles.append([self._entity2id[orig_sub], self._relation2id[pred], self._entity2id[cand]])
        possibles_tensor = torch.tensor(possibles)
        if self._cuda:
            possibles_tensor = possibles_tensor.cuda()
        out = self._kg_model(possibles_tensor)
        answers = torch.topk(out, k=k, dim=0)
        probs = answers[0]
        indexes = answers[1]
        retvals = []
        for i in range(len(indexes)):
            if orig_sub == '?':
                orig = reverse_lookup[possibles[int(indexes[i])][0]]
                triple = orig + ' ' + pred + ' ' + orig_ob
            else:
                orig = reverse_lookup[possibles[int(indexes[i])][2]]
                triple = orig_sub + ' ' + pred + ' ' + orig
            retvals.append({'prob': round(expit(float(probs[i])), 4), 'triple': triple})
        return retvals

    def _search(self, term):
        head_goal = Goal(Rule("x(y):-x(y)"))
        head_goal.rule.goals = [term]
        queue = deque([head_goal])
        iterations = 0
        max_iterations = max(100, (len(self.rules) + 1) ** 1.5)
        while queue and iterations < max_iterations:
            iterations += 1
            c = queue.popleft()
            if c.idx >= len(c.rule.goals):
                if not c.parent:
                    if c.bindings:
                        new_binding = {k:str(v) for (k, v) in c.bindings.items()}
                        yield new_binding
                    else:
                        yield True
                    continue
                parent = copy.deepcopy(c.parent)
                unify(c.rule.head, c.bindings, parent.rule.goals[parent.idx], parent.bindings)
                parent.idx += 1
                queue.append(parent)
                continue
            term = c.rule.goals[c.idx]
            pred = term.pred
            for rule in self.rules:
                if rule.head.pred != term.pred:
                    continue
                if len(rule.head.args) != len(term.args):
                    continue
                child = Goal(rule, c)
                ans = unify(term, c.bindings, rule.head, child.bindings)
                if ans:
                    queue.append(child)

    def attr(self, node_name, attributes):
        """Set attributes on an existing graph node.

        :param str node_name: Name of the node
        :param dict attributes: Dictionary of attributes to set

        :Example:

        >>> kb = KB()
        >>> kb.store('eats(tom, rice)')
        0
        >>> kb.attr('tom', {'is_person': True})
        >>> kb.node('tom')
        {'is_person': True}"""

        nx.set_node_attributes(self.G, {node_name: attributes})

    def node(self, node_name):
        """Get a node, and its attributes, from the graph.

        :param str node_name: Name of the node
        :return: The node and its attributes.

        :Example:

        >>> kb = KB()
        >>> kb.store('eats(tom, rice)')
        0
        >>> kb.node('tom')
        {}
        >>> kb.attr('tom', {'is_person': True})
        >>> kb.node('tom')
        {'is_person': True}"""

        return self.G.nodes(data=True)[node_name]

    def delete_rule(self, rule_idx):
        """Delete a rule from the KB.

        :param int rule_idx: The index of the rule in the KB. Returned when the rule was added.

        :Example:

        >>> kb = KB()
        >>> kb.store('a(a)')
        0
        >>> kb.delete_rule(0)
        True
        """
        try:
            self.rules.pop(rule_idx)
            return True
        except:
            return False

    def plot(self, density=1.0):
        """Plots a network diagram from (triple) nodes and edges in the KB.

        :param float density: Probability (0-1) that a given edge will be plotted, \
        useful to thin out dense graphs for visualization."""
        edgelist = [e for e in self.G.edges(data=True) if random.random() < density]
        newg = nx.DiGraph(edgelist)
        pos = nx.spring_layout(newg)
        plt.figure(1,figsize=(12,12))
        nx.draw_networkx_nodes(newg, pos, node_size=200)
        nx.draw_networkx_edges(newg, pos, edgelist=edgelist, width=1, font_size=8)
        nx.draw_networkx_labels(newg, pos, font_size=10, font_family='sans-serif')
        nx.draw_networkx_edge_labels(newg, pos)
        plt.axis('off')
        plt.show()

    def query(self, statement):
        """Query the KB.

        :param str statement: A rule to query on.
        :return: Generator of alternative bindings to variables that match the query

        :Example:

        >>> kb = KB()
        >>> kb.store('a(a)')
        0
        >>> kb.query('a(X)') #doctest: +ELLIPSIS
        <generator object KB._search at 0x...>
        >>> list(kb.query('a(X)'))
        [{'X': 'a'}]"""
        return self._search(Term(strip_all_whitespace(statement)))

    def store(self, statement):
        """Store a fact/rule in the KB

        :param str statement: Fact or rule to store in the KB.
        :return: the id of the fact/rule

        :Example:

        >>> KB().store('a(a)')
        0"""
        self.rules.append(Rule(strip_all_whitespace(statement), graph=self.G))
        return len(self.rules) - 1

    def to_triples(self, data=False):
        """Convert all facts in the KB to a list of triples, each of length 3
        (or 4 if data=True).
        Any fact that is not arity 2 will be ignored.

        :Note: While the Prolog style representation uses `pred(subject, object)`, \
        the triple representation is `(subject, pred, object)`.

        :param bool data: Whether to return subject attributes as a 4th element.
        :return: list of triples (tuples of length 3 or 4 if data=True)
        
        :Example:

        >>> kb = KB()
        >>> kb.store('a(b, c)')
        0
        >>> kb.to_triples()
        [('b', 'a', 'c')]
        >>> kb.store('a(a)')
        1
        >>> kb.to_triples()
        [('b', 'a', 'c')]
        >>> kb.attr('b', {'an_attribute': 'xyz'})
        >>> kb.to_triples(data=True)
        [('b', 'a', 'c', {'an_attribute': 'xyz'})]"""
        triples = []
        for r in self.rules:
            if not r.goals:
                if len(r.head.args) == 2:
                    subject = str(r.head.args[0])
                    subject = subject[0].lower() + subject[1:]
                    object_ = str(r.head.args[1])
                    object_ = object_[0].lower() + object_[1:]
                    if data:
                        triples.append((subject, r.head.pred, object_, self.node(subject)))
                    else:
                        triples.append((subject, r.head.pred, object_))
        return triples

    def from_triples(self, triples):
        """Stores facts from a list of tuples into the KB.

        :param list triples: List of tuples each of the form `(subject, pred, object)`

        :Example:
        
        >>> kb = KB()
        >>> kb.from_triples([('b', 'a', 'c')])
        >>> len(list(kb.query('a(b, c)')))
        1"""
        for (u, p, v) in triples:
            self.store('{}({},{})'.format(p, u, v))

    def from_csv(self, csvfile, header=None, start=0, size=None):
        with open(csvfile) as f:
            reader = csv.reader(f)
            i = 0
            if header:
                next(reader, None)
                i = 1
            while i < start:
                i += 1
                next(reader, None)
            i = 0
            for row in reader:
                pred = row[1].replace('.', '').replace('(', '').replace(')','')
                sub = row[0].replace(' ','').replace('.', '').replace('(', '').replace(')','')
                sub = sub[0].lower() + sub[1:]
                ob = row[2].replace(' ','').replace('.', '').replace('(', '').replace(')','')
                ob = ob[0].lower() + ob[1:]
                if not (sub.isalpha() and ob.isalpha()):
                    continue
                self.store('{}({},{})'.format(pred, sub, ob))
                i += 1
                if size and i > size:
                    break
