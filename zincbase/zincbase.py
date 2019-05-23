from collections import deque, defaultdict
import copy
import csv
import math
import os
import pickle
import random
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from logic.Goal import Goal
from logic.Negative import Negative
from logic.Term import Term
from logic.Rule import Rule
from logic.common import unify, process
from nn.dataloader import NegDataset, TrainDataset, BidirectionalOneShotIterator
from nn.rotate import KGEModel
from utils.string_utils import strip_all_whitespace, split_to_parts, cleanse

class KB():
    """Knowledge Base Class

    >>> kb = KB()
    >>> kb.__class__
    <class 'zincbase.KB'>
    """
    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.rules = []
        self._neg_examples = []
        self._entity2id = {}
        self._relation2id = {}
        self._encoded_triples = []
        self._encoded_neg_examples = []
        self._kg_model = None
        self._knn = None
        self._knn_index = []
        self._cuda = False
        self.classifiers = {}

        self._model_name = None
        self._embedding_size = None
        self._gamma = None
        self._node_attributes = None
        self._pred_attributes = None
        self._attr_loss_to_graph_loss = None
        self._pred_loss_to_graph_loss = None

    def seed(self, seed):
        """Seed the RNGs for PyTorch, NumPy, and Python itself.

        :param int seed: random seed

        :Example:

        >>> KB().seed(555)
        """
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def edge_attr(self, sub, pred, ob, attributes):
        """Set attributes on a predicate between subject and object.
        Useful for example to encode time, or truthiness.

        Note that if any of the specified attributes have been previously set,
        this updates them with new values. To delete a set edge attribute, see also `delete_edge_attr`.

        :param str sub: Subject node/entity
        :param str pred: Predicate between subject and object
        :param str ob: Object node/entity
        :param dict attributes: Attributes to set on the individual edge. Must be floats.

        :Example:

        >>> kb = KB()
        >>> kb.store('eats(tom, rice)')
        0
        >>> kb.edge_attr('tom', 'eats', 'rice', {'used_to': 1.0})
        >>> kb.edge('tom', 'eats', 'rice')
        {'used_to': 1.0}
        >>> kb.edge_attr('tom', 'eats', 'rice', {'still_does': 1.0})
        >>> kb.edge('tom', 'eats', 'rice')
        {'used_to': 1.0, 'still_does': 1.0}"""
        for idx, edge in self.G[sub][ob].items():
            if edge['pred'] == pred:
                nx.set_edge_attributes(self.G, {(sub, ob, idx): attributes})
                return None
        return False

    def delete_edge_attr(self, sub, pred, ob, attributes):
        """Delete attributes previously set on a predicate between subject and object.
        To set the attribute in the first place, see also `edge_attr`.

        :param str sub: Subject node/entity
        :param str pred: Predicate between subject and object
        :param str ob: Object node/entity
        :param list attributes: List of attributes to delete.

        :returns: False if attribute was not present, else None.
        """
        for idx, edge in self.G[sub][ob].items():
            if edge['pred'] == pred:
                for attribute in attributes:
                    del edge[attribute]
                return None
        return False

    def edge(self, sub, pred, ob):
        """Returns an edge and its attributes.

        :param str sub: Subject node/entity
        :param str pred: Predicate between subject and object
        :param str ob: Object node/entity

        :Example:

        >>> kb = KB()
        >>> kb.store('eats(tom, rice)')
        0
        >>> kb.edge_attr('tom', 'eats', 'rice', {'used_to': 1.0})
        >>> kb.edge('tom', 'eats', 'rice')
        {'used_to': 1.0}"""
        for _, edge in self.G[sub][ob].items():
            if edge['pred'] == pred:
                return {k:v for (k,v) in edge.items() if k != 'pred'}
        return False

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
                    for final_edge in pred:
                        yield path + [(pred[final_edge]['pred'], n)]
                else:
                    for edge in pred:
                        stack.append((n, depth+1, path + [(pred[edge]['pred'], n)]))
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
        """Build a classifier (SVM) for a predicate that can classify a subject, given a predicate, into
        one of the object entities from the KB that has that predicate relation. Automatically
        compensates for class imbalance.

        :Example:

        >>> kb = KB()
        >>> kb.from_csv('./assets/countries_s1_train.csv', delimiter='\\t')
        >>> kb.seed(555)
        >>> kb.build_kg_model(cuda=False, embedding_size=40)
        >>> kb.train_kg_model(steps=1000, batch_size=1, verbose=False)
        >>> _ = kb.create_multi_classifier('locatedin')
        >>> kb.multi_classify('philippines', 'locatedin')
        'south_eastern_asia'"""

        all_examples = list(self.query('{}(X, Y)'.format(pred)))
        Xs = []
        Ys = []
        indexes = list(set([x['Y'] for x in all_examples]))
        ratios = defaultdict(int)
        for example in all_examples:
            Xs.append(self.get_embedding(example['X']).cpu())
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
        """Predict `object` for subject according to the multi-classifer
        previously trained on `pred`."""

        clf, indexes = self.classifiers[pred]
        return indexes[int(clf.predict(np.reshape(self.get_embedding(subject).cpu(), (1, -1))))]

    def create_binary_classifier(self, pred, ob):
        """Creates a binary classifier (SVM) for `pred(?, ob)` using embeddings from the trained model.
        Automatically compensates for class imbalance.

        Follow it with `binary_classify(sub, pred, ob)` to predict whether the relation holds or not.

        May be useful because although the model can estimate a probability for (sub, pred, ob),
        what threshold should you use to decide what constitutes True vs False?

        :Example:

        >>> kb = KB()
        >>> kb.seed(555)
        >>> kb.from_csv('./assets/countries_s1_train.csv', delimiter='\\t')
        >>> kb.build_kg_model(cuda=False, embedding_size=100)
        >>> kb.train_kg_model(steps=2000, batch_size=1, verbose=False, neg_to_pos=4)
        >>> _ = kb.create_binary_classifier('locatedin', 'asia')
        >>> kb.binary_classify('india', 'locatedin', 'asia')
        True
        >>> kb.binary_classify('brazil', 'locatedin', 'asia')
        False"""
        all_examples = list(self.query('{}(X, Y)'.format(pred)))
        pos_examples = [self.get_embedding(x['X']) for x in all_examples if x['Y'] == ob]
        neg_examples = [self.get_embedding(x['X']) for x in all_examples if x['Y'] != ob]
        Xs = np.reshape(np.stack(pos_examples + neg_examples), (-1, pos_examples[0].shape[1]))
        Ys = np.stack([2 for x in pos_examples] + [1 for x in neg_examples])
        ratio = int(len(neg_examples) / len(pos_examples))
        clf = SVC(gamma='auto', kernel='linear', class_weight={2:min(ratio, 15)})
        clf.fit(Xs, Ys)
        self.classifiers[(pred, ob)] = clf
        return clf

    def binary_classify(self, subject, pred, ob):
        """Predict whether triple (sub, pred, ob) is true or not."""
        clf = self.classifiers[(pred, ob)]
        X = self.get_embedding(subject)
        pred = int(clf.predict(X))
        return pred == 2

    def save_all(self, dirname='.'):
        """Save current KB to the directory specified. Saves the (state dict of the) PyTorch \
        model as well, if it has been built.

        :param str dirname: Directory in which to save the files. Creates the directory \
        if it doesn't already exist."""
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if self._kg_model:
            torch.save(self._kg_model.state_dict(), os.path.join(dirname, 'pytorch_model.dict'))
        zb_dict = {
            'model_name': self._model_name,
            'entity2id': self._entity2id,
            'relation2id': self._relation2id,
            'encoded_triples': self._encoded_triples,
            'embedding_size': self._embedding_size,
            'gamma': self._gamma,
            'node_attributes': self._node_attributes,
            'pred_attributes': self._pred_attributes,
            'attr_loss_to_graph_loss': self._attr_loss_to_graph_loss,
            'pred_loss_to_graph_loss': self._pred_loss_to_graph_loss,
            'rules': self.rules
        }
        f = open(os.path.join(dirname, 'zb.pkl'), 'wb')
        pickle.dump(zb_dict, f)
        f.close()
        return True

    def load_all(self, dirname='.', cuda=False):
        """Load KB (and model, if it exists) from the specified directory.

        :param str dirname: Directory to load zb.pkl and (if present) pytorch_model.dict
        :param bool cuda: If the model exists, it will be loaded - specify if you want \
        it to be on the GPU."""

        with open(os.path.join(dirname, 'zb.pkl'), 'rb') as f:
            zb_dict = pickle.load(f)
        self._model_name = zb_dict['model_name']
        self._entity2id = zb_dict['entity2id']
        self._relation2id = zb_dict['relation2id']
        self._encoded_triples = zb_dict['encoded_triples']
        self._embedding_size = zb_dict['embedding_size']
        self._gamma = zb_dict['gamma']
        self._node_attributes = zb_dict['node_attributes']
        self._pred_attributes = zb_dict['pred_attributes']
        self._attr_loss_to_graph_loss = zb_dict['attr_loss_to_graph_loss']
        self._pred_loss_to_graph_loss = zb_dict['pred_loss_to_graph_loss']
        tmp_rules = zb_dict['rules']
        for rule in tmp_rules:
            self.store(str(rule.head))
        if os.path.exists(os.path.join(dirname, 'pytorch_model.dict')):
            self.build_kg_model(cuda, embedding_size=self._embedding_size, gamma=self._gamma,
            model_name=self._model_name, node_attributes=self._node_attributes,
            attr_loss_to_graph_loss=self._attr_loss_to_graph_loss,
            pred_loss_to_graph_loss=self._pred_loss_to_graph_loss,
            pred_attributes=self._pred_attributes)
            self._kg_model.load_state_dict(torch.load(os.path.join(dirname, 'pytorch_model.dict')))
        return True


    def build_kg_model(self, cuda=False, embedding_size=256, gamma=24, model_name='RotatE',
                    node_attributes=[], attr_loss_to_graph_loss=1.0, pred_loss_to_graph_loss=1.0,
                    pred_attributes=[]):
        """Build the dictionaries and KGE model

        :param list node_attributes: List of node attributes to include in the model. \
            If node doesn't possess the attribute, will be treated as zero. So far attributes \
        must be floats.
        :param list pred_attributes: List of predicate attributes to include in the model.
        :param float attr_loss_to_graph_loss: % to scale attribute loss against graph loss. \
        0 would only take into account graph loss, math.inf would only take into account attr loss."""
        # TODO refactor this so there's a separate dict of node + pred attrs; they don't have
        # to be part of the triple.

        self._gamma = gamma
        self._embedding_size = embedding_size
        self._model_name = model_name
        self._attr_loss_to_graph_loss = attr_loss_to_graph_loss
        self._pred_loss_to_graph_loss = pred_loss_to_graph_loss
        self._node_attributes = node_attributes
        self._pred_attributes = pred_attributes

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
            # TODO: attribute must be a float; for a dictionary encoding of them (for categoricals)
            attrs = []
            for attribute in node_attributes:
                attr = float(triple[3].get(attribute, 0.0))
                attrs.append(attr)
            for pred_attr in pred_attributes:
                if pred_attr == 'truthiness':
                    default_value = 1.
                else:
                    default_value = 0.
                attr = float(triple[4].get(pred_attr, default_value))
                attrs.append(attr)
            if len(triple) == 7 and triple[6]:
                true = 1. # it's a false fact; negative example TODO rename from 'true'!
            else:
                true = 0.
            self._encoded_triples.append((self._entity2id[triple[0]], self._relation2id[triple[1]], self._entity2id[triple[2]],
                                        attrs, true))
        for neg_example in self._neg_examples:
            self._encoded_neg_examples.append((self._entity2id[neg_example.head], self._relation2id[neg_example.pred], self._entity2id[neg_example.tail]))
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
                             pred_attributes=pred_attributes,
                             attr_loss_to_graph_loss=attr_loss_to_graph_loss,
                             pred_loss_to_graph_loss=pred_loss_to_graph_loss,
                             device=device)
        if cuda:
            self._cuda = True
            self._kg_model = self._kg_model.cuda()

    def train_kg_model(self, steps=1000, batch_size=512, lr=0.001,
                       reencode_triples=False, neg_to_pos=128,
                       neg_ratio=1., verbose=True):
        """Train a KG model on the KB.

        :param int steps: Number of training steps
        :param int batch_size: Batch size for training
        :param float lr: Initial learning rate for Adam optimizer
        :param bool reencode_triples: If a node has been added since last training, set this to True
        :param int neg_to_pos: Ratio of generated negative samples to real positive samples
        :param float neg_ratio: How often real/inputted negative examples should appear, vs real pos + generated neg. Smaller (>0) means more often.
        """
        if reencode_triples:
            # TODO: this is not encoding attributes as well, yet.
            triples = self.to_triples(data=True)
            self._encoded_triples = []
            for triple in triples:
                self._encoded_triples.append((self._entity2id[triple[0]], self._relation2id[triple[1]], self._entity2id[triple[2]]))

        nentity = len(self._entity2id)
        nrelation = len(self._relation2id)
        train_dataloader_head = DataLoader(
            TrainDataset(self._encoded_triples, nrelation, neg_to_pos, 'head-batch'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=TrainDataset.collate_fn)
        train_dataloader_tail = DataLoader(
            TrainDataset(self._encoded_triples, nrelation, neg_to_pos, 'tail-batch'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=TrainDataset.collate_fn)
        if len(self._neg_examples):
            neg_dataloader = DataLoader(
                NegDataset(self._encoded_neg_examples),
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                collate_fn=TrainDataset.collate_fn)
            neg_ratio = int(neg_ratio * (len(self._encoded_triples) / len(self._neg_examples)))
            neg_ratio = max(neg_ratio, 1e-4)
            train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail, neg_dataloader, neg_ratio)
        else:
            train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._kg_model.parameters()), lr=lr)

        self._kg_model.train()
        if verbose:
            it = tqdm(range(0, steps))
        else:
            it = range(0, steps)
        for step in it:
            log = self._kg_model.train_step(self._kg_model, optimizer, train_iterator, {'cuda': self._cuda})
            if verbose and step % 100 == 0:
                print(log)
        self._kg_model.eval()

    def estimate_triple_prob(self, sub, pred, ob):
        """Estimate the probability of the triple (sub, pred, ob) according to the trained model."""

        # TODO: Should be prolog style
        if not self._kg_model:
            raise Exception('Must build and train the model first')
        tensor = torch.tensor([[self._entity2id[sub], self._relation2id[pred], self._entity2id[ob]]])
        if self._cuda:
            tensor = tensor.cuda()
        logit, _ = self._kg_model(tensor, attributes=False, predict_only=True)
        return round(expit(float(logit)), 4)

    def estimate_triple_prob_with_attrs(self, sub, pred, ob, pred_prop):
        # TODO: Should be prolog style
        if not self._kg_model:
            raise Exception('Must build and train the model first')
        tensor = torch.tensor([[self._entity2id[sub], self._relation2id[pred], self._entity2id[ob]]])
        if self._cuda:
            tensor = tensor.cuda()
        logit, _ = self._kg_model(tensor, attributes=True, predict_pred_prop=pred_prop, predict_only=True)
        return round(expit(float(logit)), 4)

    def get_embedding(self, entity):
        index = torch.LongTensor([self._entity2id[entity]])
        if self._cuda:
            index = index.cuda()
        return torch.index_select(self._kg_model.entity_embedding, dim=0, index=index).detach()

    def fit_knn(self, entities=None):
        """Fit an unsupervised sklearn kNN to the embeddings of entities.

        :param list entities: The entities that should be part of the kNN. Defaults to all if not specified"""
        self._knn_index = []
        if not entities:
            entities = [e for e in self._entity2id]
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
        """Get the nearest neighbors to entity (embedding), according to the previously fit knn.

        :param str entity: An entity
        :param int k: How many neighbors
        """
        embedding = self.get_embedding(entity)
        embedding = embedding.cpu() # no cuda for sklearn
        distances, indices = self._knn.kneighbors(embedding, n_neighbors=k)
        borgs = []
        distances = distances[0]
        indices = indices[0]
        for i in range(len(distances)):
            borgs.append({'distance': round(distances[i], 4), 'entity': self._knn_index[int(indices[i])]})
        return borgs

    @property
    def entities(self):
        """All the entities in the KB.

        :returns generator: Generator of all the entities"""
        return self._entity2id.keys()

    @property
    def predicates(self):
        """All the predicates (aka relations) in the KB.

        :returns generator: Generator of all the predicates"""
        return self._relation2id.keys()

    def get_most_likely(self, sub, pred, ob, candidates=None, k=1):
        """Return the k most likely triples to satisfy the input triple. One of \
        sub, pred, or ob may be '?'.

        :param list<str> candidates: Candidate entities/predicates. If None or not specified, this function \
        will generate possible candidates from the rest of the triple.
        :param int k: The k in top k.

        :Example:

        >>> kb = KB()
        >>> kb.from_csv('./assets/countries_s1_train.csv', delimiter='\\t')
        >>> kb.seed(555)
        >>> kb.build_kg_model(cuda=False, embedding_size=100)
        >>> kb.train_kg_model(steps=2000, batch_size=2, verbose=False, neg_to_pos=4)
        >>> kb.get_most_likely('austria', 'neighbor', '?', k=2) # doctest:+ELLIPSIS
        [{'prob': 0.9673, 'triple': ('austria', 'neighbor', 'germany')}, {'prob': 0.9656, 'triple': ('austria', 'neighbor', 'liechtenstein')}]
        >>> kb.get_most_likely('?', 'neighbor', 'austria', candidates=list(kb.entities), k=2)
        [{'prob': 0.9467, 'triple': ('slovenia', 'neighbor', 'austria')}, {'prob': 0.94, 'triple': ('liechtenstein', 'neighbor', 'austria')}]
        >>> kb.get_most_likely('austria', '?', 'germany', k=3)
        [{'prob': 0.9673, 'triple': ('austria', 'neighbor', 'germany')}, {'prob': 0.664, 'triple': ('austria', 'locatedin', 'germany')}]"""
        
        reverse_lookup = {}
        possibles = []
        orig_sub = sub
        orig_ob = ob
        if not candidates:
            if pred == '?':
                candidates = self.predicates
            else:
                if sub == '?':
                    sub = 'X'
                    ob = 'Y'
                else:
                    ob = 'X'
                    sub = 'Y'
                candidates = self.query('{}({}, {})'.format(pred, sub, ob))
                candidates = list(set([x['X'] for x in candidates]))
        for cand in candidates:
            if pred == '?':
                reverse_lookup[self._relation2id[cand]] = cand
                possibles.append([self._entity2id[sub], self._relation2id[cand], self._entity2id[ob]])
            else:
                reverse_lookup[self._entity2id[cand]] = cand
                if orig_sub == '?':
                    possibles.append([self._entity2id[cand], self._relation2id[pred], self._entity2id[orig_ob]])
                else:
                    possibles.append([self._entity2id[orig_sub], self._relation2id[pred], self._entity2id[cand]])
        possibles_tensor = torch.tensor(possibles)
        if self._cuda:
            possibles_tensor = possibles_tensor.cuda()
        out, _ = self._kg_model(possibles_tensor, predict_only=True)
        k = min(out.size(0), k)
        answers = torch.topk(out, k=k, dim=0)
        probs = answers[0]
        indexes = answers[1]
        retvals = []
        for i in range(len(indexes)):
            if pred == '?':
                orig = reverse_lookup[possibles[int(indexes[i])][1]]
                triple = (sub, orig, ob)
            elif orig_sub == '?':
                orig = reverse_lookup[possibles[int(indexes[i])][0]]
                triple = (orig, pred, orig_ob)
            else:
                orig = reverse_lookup[possibles[int(indexes[i])][2]]
                triple = (orig_sub, pred, orig)
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

    def delete_rule(self, rule_idx):
        """Delete a rule from the KB.

        :param rule_idx: The index of the rule in the KB. Returned when the rule was added. May be int (if it \
        was a real rule) or str (if it was a negative example - preceded by ~).

        :Example:

        >>> kb = KB()
        >>> kb.store('a(a)')
        0
        >>> kb.delete_rule(0)
        True
        """
        try:
            if isinstance(rule_idx, str) and rule_idx[0] == '~':
                rule_idx = int(rule_idx[1:])
                self._neg_examples.pop(rule_idx)
                return True
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

    def solidify(self, predicate):
        """Query the KB (with Prolog) and 'solidify' facts in the KB, making them part
        of the graph, so that the NN can be trained.

        :param str predicate: A predicate (that's a rule not a fact otherwise what's the point)

        :Example:

        >>> kb = KB()
        >>> kb.store('is(tom, human)')
        0
        >>> kb.store('has_part(shamala, head)')
        1
        >>> kb.store('is(X, human) :- has_part(X, head)')
        2
        >>> next(kb.query('is(tom, human)'))
        True
        >>> kb.to_triples()
        [('tom', 'is', 'human'), ('shamala', 'has_part', 'head')]
        >>> kb.solidify('is')
        1
        >>> kb.to_triples()
        [('tom', 'is', 'human'), ('shamala', 'has_part', 'head'), ('shamala', 'is', 'human')]
        """
        answers = self.query('{}(X, Y)'.format(predicate))
        i = 0
        rule_strings = [str(x) for x in self.rules]
        for a in answers:
            as_string = '{}({}, {})'.format(predicate, a['X'], a['Y'])
            if not as_string in rule_strings:
                i += 1
                self.store(as_string)
        return i

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

    def store(self, statement, node_attributes=[], edge_attributes={}):
        """Store a fact/rule in the KB

        It is possible to store 'false' facts (negative examples) by preceding the predicate with a tilde (~).
        In this case, they do not come out in the graph and cannot be queried, but may
        assist when building the model.

        :param str statement: Fact or rule to store in the KB.
        :param list<dict> node_attributes: List of length 2 with each element being a \
        dict of items to set on the nodes (in order subject, object).
        :param dict edge_attributes: Dictionary of attributes to set on the edge. May \
        include truthiness which, if < 0, automatically makes the rule a negative example.
        :return: the id of the fact/rule

        :Example:

        >>> KB().store('a(a)')
        0"""
        statement = strip_all_whitespace(statement)
        if 'truthiness' in edge_attributes and edge_attributes['truthiness'] < 0:
            if statement[0] != '~':
                statement = '~' + statement
        if statement[0] == '~':
            triple = split_to_parts(statement[1:])
            if not triple[0] in self._entity2id:
                self._entity2id[triple[0]] = len(self._entity2id)
            if not triple[1] in self._relation2id:
                self._relation2id[triple[1]] = len(self._relation2id)
            if not triple[2] in self._entity2id:
                self._entity2id[triple[2]] = len(self._entity2id)
            self._neg_examples.append(Negative(statement[1:]))
            return '~' + str(len(self._neg_examples) - 1)
        self.rules.append(Rule(statement, graph=self.G))
        if edge_attributes:
            parts = split_to_parts(statement)
            self.edge_attr(parts[0], parts[1], parts[2], edge_attributes)
        if node_attributes:
            parts = split_to_parts(statement)
            self.attr(parts[0], node_attributes[0])
            self.attr(parts[2], node_attributes[1])
        return len(self.rules) - 1

    def to_tensorboard_projector(self, embeddings_filename, labels_filename, filter_fn=None):
        """Convert the KB's trained embeddings to 2 files suitable for \
        https://projector.tensorflow.org. This outputs only entity embeddings, \
        not relation embeddings, a visualization of which may not be interpretable.

        :param str embeddings_filename: Filename to output embeddings to, tsv format.
        :param str labels_filename: Filename to output labels to, one label per row.
        :param function filter_fn: Only include the embeddings/labels for which filter_fn(label) returns True"""

        embeddings = self._kg_model.entity_embedding.detach().cpu().numpy()
        if not filter_fn:
            filter_fn = lambda x: True
        z = [e for e in zip(embeddings, self.entities) if filter_fn(e[1])]
        embeddings, labels = list(zip(*z))
        np.savetxt(embeddings_filename, embeddings, delimiter='\t')
        labels_file = open(labels_filename, 'w')
        for ent in labels:
            labels_file.write(ent + '\n')
        labels_file.close()
        return True

    def to_triples(self, data=False):
        """Convert all facts in the KB to a list of triples, each of length 3
        (or 4 if data=True).
        Any fact that is not arity 2 will be ignored.

        :Note: While the Prolog style representation uses `pred(subject, object)`, \
        the triple representation is `(subject, pred, object)`.

        :param bool data: Whether to return subject, predicate and object \
        attributes as elements 4, 5, and 6 of the triple. The 7th element of the \
        triple is usually False, but is True when the fact/triple is a negative example.
        :return: list of triples (tuples of length 3 or 7 if data=True)

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
        >>> kb.to_triples()
        [('b', 'a', 'c')]
        >>> kb.to_triples(data=True)
        [('b', 'a', 'c', {'an_attribute': 'xyz'}, {}, {}, False)]"""
        triples = []
        neg_examples = [str(x) for x in self._neg_examples]
        for r in self.rules:
            if not r.goals:
                if len(r.head.args) == 2:
                    subject = str(r.head.args[0])
                    subject = subject[0].lower() + subject[1:]
                    object_ = str(r.head.args[1])
                    object_ = object_[0].lower() + object_[1:]
                    if data:
                        edge = self.edge(subject, r.head.pred, object_)
                        truthiness = edge.get('truthiness', False)
                        if (truthiness and truthiness < 0) or str(r) in neg_examples:
                            is_neg = True
                        else:
                            is_neg = False
                        triples.append((subject, r.head.pred, object_,
                            self.node(subject),
                            edge,
                            self.node(object_),
                            is_neg
                        ))
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

    def from_csv(self, csvfile, header=None, start=0, size=None, delimiter=','):
        with open(csvfile) as f:
            reader = csv.reader(f, delimiter=delimiter)
            i = 0
            if header:
                next(reader, None)
                i = 1
            while i < start:
                i += 1
                next(reader, None)
            i = 0
            for row in reader:
                pred = cleanse(row[1])
                sub = cleanse(row[0])
                ob = cleanse(row[2])
                if not (sub.replace('_','').isalnum() and ob.replace('_','').isalnum()):
                    continue
                node_attributes = [
                    {'_': {'real_name': row[0]}},
                    {'_': {'real_name': row[2]}}
                ]
                edge_attributes = {
                    '_': {'real_name': row[1]}
                }
                self.store('{}({},{})'.format(pred, sub, ob), node_attributes=node_attributes, edge_attributes=edge_attributes)
                i += 1
                if size and i > size:
                    break
