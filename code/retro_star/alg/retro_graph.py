import pickle
from queue import Queue
import numpy as np
from retro_star.alg.syn_route import SynRoute
from retro_star.common import args, prepare_starting_molecules, prepare_mlp, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.alg.train_policy_gnn import Model
from tqdm import tqdm
from mlp_retrosyn.mlp_inference import MLPModel

import torch
import pickle

from multiprocessing import Pool
import copy

from itertools import product

from rdkit import Chem
from rdkit.Chem import AllChem

import pandas as pd
from graphviz import Digraph

from sklearn.cluster import KMeans

from torch_geometric.data import Data, Batch
import os

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class Node(object):
    def __init__(self, cost:float, future_cost:float):
        self.parents = []
        self.children = []
        self.cost = cost
        self.future_cost = future_cost
        self.open = True
        self.success = False
        self.cost_backup = None # used when mask cost

class MolNode(Node):
    def __init__(self, smiles: str, cost:float=0, future_cost:float=0):
        super().__init__(cost, future_cost)
        self.smiles = smiles
    
    def __hash__(self) -> int:
        return hash(self.smiles)

class RecNode(Node):
    def __init__(self, template: str, reaction_cost:float, cost: float, future_cost: float):
        super().__init__(cost, future_cost)
        self.reaction_cost = reaction_cost # the cost of this single reaction
        self.template = template
        self.open = False

class RouteTree(object):
    def __init__(self, root: MolNode):
        self.root = MolNode(root.smiles)

    def merge(self, rec:RecNode, children):
        rec_cpy = RecNode(rec.template, rec.reaction_cost, rec.cost, rec.future_cost)
        self.root.parents = [rec_cpy]
        rec_cpy.parents = [self.root]
        for child in children:
            rec_cpy.children.append(child.root)
            child.root.parents.append(rec_cpy)

    def _all_recnodes(self):
        queue = Queue()
        visited = set()
        ret = []
        queue.put(self.root)
        visited.add(self.root)
        while not queue.empty():
            top = queue.get()
            if type(top) == RecNode:
                ret.append(top)
            for child in top.children:
                if not child in visited:
                    queue.put(child)
                    visited.add(child)
        return ret

    def len_and_cost(self):
        rec_nodes = self._all_recnodes()
        return len(rec_nodes), sum(v.reaction_cost for v in rec_nodes)

class GraphData(object):
    def __init__(self):
        super().__init__()

        self.x = []
        self.x_feature = []
        self.edge_index = [[], []]
        self.edge_attr = []
    
    def to_data(self):
        return Data(
            x = torch.Tensor(np.array(self.x)),
            x_feature = torch.Tensor(self.x_feature),
            edge_index = torch.Tensor(self.edge_index).long(),
            edge_attr = torch.Tensor(self.edge_attr)
        )

class SearchGraph(object):
    def __init__(self, target_mols, known_mols, use_gnn:bool=False, gnn_ckpt: str='', gnn_ratio:float=1, gnn_args=None):
        self.target_mols = target_mols
        self.known_mols = known_mols

        self.smiles_to_node = {}
        self.graph_data = GraphData()
        self.node_open = [] #
        self.nodes = [] # id2node
        self.node2id = {}

        for target_mol in target_mols:
            self.add_mol(target_mol, 0)

        self.use_gnn = use_gnn

        if use_gnn:
            self.gnn = Model.load_from_checkpoint(gnn_ckpt, **gnn_args)
            self.gnn.eval()
        else:
            self.gnn = None
        
        self.gnn_ratio=gnn_ratio


    @property
    def success(self):
        return self.success_count == len(self.target_mols)

    @property
    def success_count(self):
        return sum(1 for target_mol in self.target_mols if self.smiles_to_node[target_mol].success)

    def is_known(self, smiles):
        return smiles in self.known_mols

    def add_mol(self, smiles, cost)->Node:
        if smiles not in self.smiles_to_node:
            node = MolNode(smiles)
            node.cost = cost
            if self.is_known(smiles):
                node.open = False
                node.success = True
            self.smiles_to_node[smiles] = node

            # update graph for GNN
            self.nodes.append(node)
            self.node2id[node] = len(self.nodes)-1
            self.graph_data.x.append(smiles_to_fp(node.smiles)) # No pack
            self.graph_data.x_feature.append([
                0, # type =mol,
                1 if node.success else 0,
                1 if node.smiles in self.target_mols else 0,
                0, # reaction cost
                cost
            ])
            self.node_open.append(node.open)

        if cost < self.smiles_to_node[smiles].cost:
            # self.push_down_cost(self.smiles_to_node[smiles], self.smiles_to_node[smiles].cost - cost)
            self.push_down_cost_mol(self.smiles_to_node[smiles], cost)
        return self.smiles_to_node[smiles]

    
    def push_down_cost_success_inf(self, node: MolNode):
        assert node.success
        inf = float('inf')
        if node.cost == inf:
            return
        
        node.cost_backup = node.cost
        node.cost = inf

        for reaction in node.children:
            reaction.cost_backup = reaction.cost
            reaction.cost = inf
            for grand in reaction.children:
                if grand.parents and min(p.cost for p in grand.parents) == inf: # all parents success
                    self.push_down_cost_success_inf(grand)
    
    def push_down_cost_mol(self, node: MolNode, cost: float, visited=None):
        if visited is None:
            visited = set()
        
        if node in visited:
            return

        visited.add(node)

        node.cost = cost
        self.graph_data.x_feature[self.node2id[node]][4] = cost
        for reaction in node.children:
            reaction.cost = reaction.reaction_cost + node.cost
            for grand in reaction.children:
                if reaction.cost < grand.cost:
                    self.push_down_cost_mol(grand, reaction.cost, visited)

    def add_reaction(self, reactants, template: str, reaction_cost: float, cost:float):
        node = RecNode(template, reaction_cost=reaction_cost, cost=cost, future_cost=0)

        rec_id = len(self.nodes)
        self.nodes.append(node)
        self.node2id[node] = rec_id
        self.graph_data.x.append([0]*2048)
        self.graph_data.x_feature.append([
            1,
            1 if node.success else 0,
            0,
            reaction_cost,
            cost
        ])
        self.node_open.append(False)
        
        for i in range(len(reactants)):
            child = self.add_mol(reactants[i], cost)
            child.parents.append(node)
            node.children.append(child)

            child_id = self.node2id[child]

            self.graph_data.edge_index[0].extend([rec_id, child_id])
            self.graph_data.edge_index[1].extend([child_id, rec_id ])
            self.graph_data.edge_attr.extend([0, 1])

        return node


    def expand(self, current, reactant_lists, template_list, costs):
        assert type(current) == MolNode
        assert current.open
        assert not current.children
        assert len(reactant_lists) == len(costs)

        new_success_mols = set() # used to push up success status
        new_mols = set() # used to push up future cost

        current_id = self.node2id[current]
        for i in range(len(reactant_lists)):
            if any(len(c) > 300 for c in reactant_lists[i]):
                costs[i] = 100
            child_rec = self.add_reaction(reactant_lists[i], template_list[i], costs[i], costs[i] + current.cost)
            child_rec.parents.append(current)
            current.children.append(child_rec)
            child_id = self.node2id[child_rec]
            self.graph_data.edge_index[0].extend([current_id, child_id])
            self.graph_data.edge_index[1].extend([child_id, current_id])
            self.graph_data.edge_attr.extend([0, 1])

            for m in child_rec.children:
                new_mols.add(m)
                if m.success:
                    new_success_mols.add(m)

        return new_success_mols, new_mols

    def push_up_success(self, current):
        assert current.success
        self.graph_data.x_feature[self.node2id[current]][1] = 1
        for parent in current.parents:
            if type(parent) == MolNode: # OR node
                if not parent.success:
                    parent.success = True
                    self.push_up_success(parent)
            else: # AND (mol) node
                if not parent.success and all(m.success for m in parent.children) and ( not parent.parents[0].success):
                    parent.success = True
                    self.push_up_success(parent)
    
    def push_up_future_cost(self, current, visited=None):

        if visited is None:
            visited = set()
        if current in visited:
            return
        visited.add(current)

        if len(visited) > 500:
            return

        for parent in current.parents:
            if type(parent) == RecNode:
                new_parent_future_cost = parent.reaction_cost + sum(m.future_cost for m in parent.children)
                if parent.future_cost != new_parent_future_cost:
                    parent.future_cost = new_parent_future_cost
                    self.push_up_future_cost(parent, visited)
            else:
                new_parent_future_cost = min(m.future_cost for m in parent.children)
                if new_parent_future_cost != parent.future_cost:
                    parent.future_cost = new_parent_future_cost
                    self.push_up_future_cost(parent, visited)


    def get_estimated_future_cost(self, current):

        node = current
        ret = 0
        visted_rec = set()
        while True:
            if not node.parents:
                break
            parent_id = np.argmin(p.cost for p in node.parents)
            parent_rec = node.parents[parent_id]
            if parent_rec in visted_rec:
                break

            visted_rec.add(parent_rec)

            other_cost = sum((p.future_cost for p in parent_rec.children if p != node))
            # other_cost = parent_rec.future_cost - parent_rec.reaction_cost - node.future_cost
            assert other_cost>=0
            ret += other_cost
            node = parent_rec.parents[0]
        return ret
    
    def find_next_to_expand_gnn(self):
        assert self.gnn is not None

        with torch.no_grad():
            batch = Batch.from_data_list([self.graph_data.to_data()])

        pred = self.gnn(batch)
        open_nodes = torch.BoolTensor(self.node_open, device=pred.device)
        pred[~open_nodes] = -float('inf') # don't use closed nodes
        return pred
        

    def find_next_to_expand(self):
        if self.use_gnn: 
            gnn_pred = self.find_next_to_expand_gnn()
            gnn_pred = torch.softmax(gnn_pred, dim=0)
        else:
            gnn_pred = torch.zeros((len(self.node2id)))
        ret = None
        best_cost = 1e8
        for m in self.smiles_to_node.values():
            if m.open:
                if not self.use_gnn:
                    total_cost = m.cost + self.get_estimated_future_cost(m)
                else:
                    total_cost = m.cost - gnn_pred[self.node2id[m]]*self.gnn_ratio
                if ret is None or total_cost < best_cost:
                    ret = m
                    best_cost = total_cost
        return ret
    
    def get_success_routes(self, topk=2):
        assert self.success

        def dfs(root: MolNode, visited: set):
            assert root.success

            ret = []
            visited.add(root)
            success_reaction = []
            for reaction in root.children:
                if reaction.success:
                    if any(m.smiles in visited for m in reaction.children):
                        continue # avoid loop
                    else:
                        success_reaction.append(reaction)
            success_reaction = sorted(success_reaction, key=lambda r: r.future_cost)

            for reaction in success_reaction[:topk]:
                child_mol_route = []
                for child_mol in reaction.children:
                    child_mol_route.append(dfs(child_mol, copy.deepcopy(visited)))
                
                for r in product(*child_mol_route):
                    route = SynRoute(
                        target_mol = root.smiles,
                        succ_value=0,
                        search_status=0
                    )
                    route.set_value(root.smiles, reaction.reaction_cost + sum(rr.values[0] for rr in r))


        syn_route = SynRoute(
            target_mol=self.target_mol,
            succ_value=0,
            search_status=0
        )

        mol_queue = Queue()
        mol_queue.put((self.smiles_to_node[self.target_mol], syn_route))

        all_routes = [syn_route]

        while not mol_queue.empty():
            mol, route = mol_queue.get()
            if self.is_known(mol.smiles):
                route.set_value(mol.smiles, 0)
                continue
            else:
                success_reaction = []
                for reaction in mol.children:
                    if reaction.success:
                        success_reaction.append(reaction)
                success_reaction = sorted(success_reaction, key=lambda r: r.future_cost)

                route_origin = copy.deepcopy(route)
                for rec_idx, reaction in enumerate(success_reaction[:topk]):
                    if rec_idx > 0: # split
                        new_route = copy.deepcopy(route_origin)
                        all_routes.append(new_route)
                    else:
                        new_route = route

                    for reacant in reaction.children:
                        mol_queue.put((reacant, new_route))

                    new_route.add_reaction(
                        mol = mol.smiles,
                        value = reaction.future_cost,
                        template=reaction.template,
                        reactants=list((m.smiles for m in reaction.children)),
                        cost = reaction.reaction_cost
                    )

        return all_routes

    def get_best_route_dfs(self, target):
        assert self.success
        # self.vis_search()

        def dfs(node: MolNode, visited: set):
            assert node.success

            visited.add(node)

            node_tree = RouteTree(node)

            if not node.children: # leaf
                visited.remove(node)
                return (node_tree, 0)

            best_subtree = None
            best_rec = None
            best_subtree_cost = 1e8

            node.children = [rec for rec in node.children if rec.success]
            for rec in node.children:
                if not any(chmol in visited for chmol in rec.children):
                    child_cost = rec.reaction_cost
                    subsubtrees = []
                    for chmol in rec.children:
                        if child_cost > best_subtree_cost:
                            break
                        subsubtree, subsubcost = dfs(chmol, visited)
                        if subsubtree is not None:
                            child_cost += subsubcost
                            subsubtrees.append(subsubtree)
                        else:
                            break
                    if len(rec.children) != len(subsubtrees): # loop
                        continue
                    if child_cost < best_subtree_cost:
                        best_subtree = subsubtrees
                        best_rec = rec
                        best_subtree_cost = child_cost
            
            if best_subtree is None:
                visited.remove(node)
                return (None, 0)
            else:
                node_tree.merge(best_rec, best_subtree)
                visited.remove(node)
                return (node_tree, best_subtree_cost)

        ret = dfs(self.smiles_to_node[target], set())
        assert ret[0] is not None
        return ret[0]

    def vis_search(self):
        G = Digraph('G', filename='search.pdf')
        G.attr(rankdir='LR')
        G.attr('node', shape='box')
        G.format = 'pdf'

        for smiles, node in self.smiles_to_node.items():
            if node.success:
                G.node(name=smiles, label=smiles)
                for rec in node.children:
                    if rec.success:
                        name = str(id(rec))[-6:]
                        G.node(name=name, label=name)
            
        for smiles, node in self.smiles_to_node.items():
            if node.success:
                for rec in node.children:
                    if rec.success:
                        rec_name = name = str(id(rec))[-6:]
                        G.edge(smiles, rec_name)
                        for cm in rec.children:
                            G.edge(rec_name, cm.smiles)
        
        G.render()

    def get_best_route(self):
        assert self.success
        syn_route = SynRoute(
            target_mol=self.target_mol,
            succ_value=0,
            search_status=0
        )

        mol_queue = Queue()
        mol_queue.put(self.smiles_to_node[self.target_mol])
        visited_reaction = set() # TODO: why we have loop here?
        while not mol_queue.empty():
            mol = mol_queue.get()
            if self.is_known(mol.smiles):
                syn_route.set_value(mol.smiles, 0)
                continue

            for reaction in mol.children:
                if reaction.success: # TODO: find best
                    if id(reaction) in visited_reaction:
                        break # already added
    
                    for reactant in reaction.children:
                        mol_queue.put(reactant)
                    syn_route.add_reaction(
                        mol = mol.smiles,
                        value=0,
                        template=reaction.template,
                        reactants=list((m.smiles for m in reaction.children)),
                        cost = mol.cost
                    )
                    visited_reaction.add(id(reaction)) # avoid loop
                    break
            else:
                raise Exception("No success route")

        return syn_route

    def debug(self):
        for mol in self.smiles_to_node.values():
            if mol.parents:
                assert np.fabs(mol.cost -  min(p.cost for p in mol.parents)) < 1e-6 or mol.cost < min(p.cost for p in mol.parents)
            if mol.children:
                assert np.fabs(mol.future_cost - min(r.future_cost for r in mol.children)) < 1e-6
                assert mol.success == any(r.success for r in mol.children)

            for rec in mol.children:
                assert len(rec.parents) == 1
                assert not mol.open
                assert np.fabs(rec.cost - rec.parents[0].cost - rec.reaction_cost) < 1e-6
                assert np.fabs(rec.future_cost - sum(r.future_cost for r in rec.children) - rec.reaction_cost) < 1e-6
                assert rec.success == all(r.success for r in rec.children)
    

    def search(self, iterations, expand_fn, value_fn, max_succes_count=1):
        first_iter = iterations
        success_count = 0

        for i in range(iterations):
            mol = self.find_next_to_expand()
            if not mol: # No open
                break

            result = expand_fn.run(mol.smiles, topk=50)
            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                scores = result['scores']
                costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                # costs = 1.0 - np.array(scores)
                if 'templates' in result.keys():
                    templates = result['templates']
                else:
                    templates = result['template']

                reactant_lists = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)
                
                new_success_mols, new_mols = self.expand(mol, reactant_lists, templates, costs)
                for new_mol in new_success_mols:
                    self.push_up_success(new_mol)

                for new_mol in new_mols:
                    # new mol may not new, it can also point to an existing one
                    # also not update if point to root
                    if len(new_mol.parents) == 1 and new_mol.smiles not in self.target_mols and not self.use_gnn:
                        new_mol.future_cost = value_fn(new_mol.smiles)
                    self.push_up_future_cost(new_mol)

                mol.open = False
                self.node_open[self.node2id[mol]]=False

                if self.success:
                    success_count += 1
                    if success_count == 1:
                        first_iter = i+1
                    if success_count >= max_succes_count:
                        break
                    
                    for target in self.target_mols:
                        target_node = self.target_mols[target]
                        if target_node.success:
                            self.push_down_cost_success_inf(target_node)
            else: # no template
                mol.open = False
                self.node_open[self.node2id[mol]]=False
                mol.future_cost = float('inf')
                self.push_up_future_cost(mol)

        n_mol_nodes = 0
        n_rec_nodes = 0
        for node in self.smiles_to_node.values():
            n_mol_nodes += 1
            n_rec_nodes += len(node.children)
        
        routes = []
        lengths = []
        costs = []
        if self.success:
            for target in self.target_mols:
                route = self.get_best_route_dfs(target)
                length, cost = route.len_and_cost()

                routes.append(route)
                lengths.append(length)
                costs.append(cost)

        return {
            'success': self.success,
            'success_count': self.success_count,
            'iter': first_iter,
            'bsz': len(self.target_mols),
            'avg_iter': first_iter / len(self.target_mols), # avege iter per target
            'n_mol_nodes': n_mol_nodes,
            'n_rec_nodes': n_rec_nodes,
            'routes': routes,
            'lengths': lengths,
            'costs': costs,
        }

def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=bool)
    arr[onbits] = 1

    if pack:
        arr = np.packbits(arr)

    return arr

def convert_result(tree):
    data = {}
    if not tree.success:
        return data

    rec_filter_thres = -np.log(0.8)

    for smiles, node in tree.smiles_to_node.items():
        if not node.success:
            continue
        
        if not node.children: # leaf
            data[smiles] = {
                'value': 0,
                'fps': smiles_to_fp(smiles, pack=True),
                'template': '',
                'children': '',
                'negative': []
            }
            continue
        
        best_rec = None
        best_cost = 100000
        for rec in node.children:
            if rec.success and rec.reaction_cost < rec_filter_thres and (best_rec is None or best_cost > rec.future_cost):
                best_rec = rec
                best_cost = rec.future_cost
        
        if best_rec is None:
            continue
        # assert best_rec.future_cost == node.future_cost
        assert smiles not in data

        negative = []

        for rec in node.children:
            if not rec.success and 0<len(rec.children)<=3 :
                item = {
                    'reaction_cost': rec.reaction_cost,
                    'target_value' : node.future_cost,
                    'reactant_fps': [smiles_to_fp(m.smiles, pack=True) for m in rec.children] + [None] * (3-len(rec.children)),
                    'reactant_masks': [1] * len(rec.children) + [0]*(3-len(rec.children)),
                    'reactant_smiles': [m.smiles for m in rec.children] + [''] * (3-len(rec.children)),
                }
                
                assert item['reactant_fps'][0] is not None
                for i in range(3):
                    if item['reactant_fps'][i] is None:
                        item['reactant_fps'][i] = item['reactant_fps'][0]

                assert len(item['reactant_fps']) == 3
                assert len(item['reactant_masks']) == 3
                assert len(item['reactant_smiles']) == 3
                negative.append(item)
            if len(negative) > 15:
                break

        data[smiles] = {
            'value': node.future_cost,
            'fps': smiles_to_fp(smiles, pack=True),
            'template': best_rec.template,
            'children': '.'.join(r.smiles for r in best_rec.children),
            'negative': negative
        }
    
    return data

def search_worker(func_args):
    route, use_value_fn, max_succes_count, collrect_data, args = func_args
    if not hasattr(search_worker, 'starting_mols'):
        search_worker.starting_mols = prepare_starting_molecules(args.starting_molecules)

    if not hasattr(search_worker, 'one_step'):
        search_worker.one_step = prepare_mlp(args.mlp_templates, args.mlp_model_dump, gpu=args.gpu)

    if not hasattr(search_worker, 'value_fn'):
        if use_value_fn:
            device = torch.device('cpu')
            model = ValueMLP(
                n_layers=1,
                fp_dim=2048,
                latent_dim=128,
                dropout_rate=0.1,
                device=device
            ).to(device)
            model_f = args.value_model # '%s/%s' % ('saved_models', 'best_epoch_final_4.pt')
            model.load_state_dict(torch.load(model_f,  map_location=device))
            model.eval()
            def value_fn(mol):
                fp = smiles_to_fp(mol, fp_dim=2048).reshape(1,-1)
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v
            search_worker.value_fn = value_fn
        else:
            def value_fn(mol):
                return 0
            
            search_worker.value_fn = value_fn

    target_mols = [r[0].split('>')[0] for r in route]
    assert len(target_mols) > 0
    gnn_args = {
        'dim': args.gnn_dim,
        'dropout': args.gnn_dropout,
        'n_layers': args.gnn_layers,
        'lr': 0 # no finetune
    }
    graph = SearchGraph(target_mols, search_worker.starting_mols, args.use_gnn_plan, args.gnn_ckpt, args.gnn_ratio, gnn_args)
    total_iter = args.iterations * len(target_mols)
    search_ret = graph.search(iterations=total_iter, expand_fn=search_worker.one_step, value_fn=search_worker.value_fn, max_succes_count=max_succes_count)
    if collrect_data:
        data = convert_result(graph)
    else:
        data = None
    
    return data, search_ret


def aug_gold_rxn(args, gold_rxn_list, aug_thr):
    device = 0 if args.gpu > -1 else -1

    forward_model = MLPModel(args.forward_model, args.mlp_templates, device=device)

    if args.fw_backward_validate:
        backward_model = MLPModel(args.fw_backward_model, args.mlp_templates, device=device)

    aug_rxn_list = []
    cnt_none = 0
    for gold_rxn in tqdm(gold_rxn_list):
        rxn, tpl = gold_rxn
        react, _, prod = rxn.split(">")
        forward_result = forward_model.run(react, topk=args.aug_topk, backward=False)

        if forward_result is None:
            cnt_none += 1
            continue
        else:
            predictions_list = forward_result['reactants']  # Actually, this it predicted product
            scores_list = forward_result['scores']
            template_list = forward_result['template']

            costs_list = [0.0 - np.log(np.clip(np.array(score), 1e-3, 1.0)) for score in scores_list]

            for pred, cost, pred_tpl in zip(predictions_list, costs_list, template_list):
                if cost > aug_thr:
                    break
                if args.fw_backward_validate:
                    backward_result = backward_model.run(pred, topk=1)

                    if backward_result is None or pred_tpl not in backward_result['template']:
                        continue

                aug_rxn_list.append((react + '>>' + pred, pred_tpl))

    gold_rxn_list.extend(aug_rxn_list)

    return gold_rxn_list


def make_batch(data, bsz):
    print('make batch', len(data), bsz)
    for i in range(0, len(data), bsz):
        yield data[i: i+bsz]


def make_test_batch(data, method, bsz, n_clusters):
    if method == 'none':
        print('Sequential batch', len(data), bsz)
        for b in make_batch(data, bsz):
            yield b
    elif method == 'pregroup':
        cluster_size = len(data) // n_clusters
        print('pregroup', cluster_size)
        for cid in range(0, len(data), cluster_size):
            cluster_data = data[cid: cid+cluster_size]
            for i in range(0, len(cluster_data), bsz):
                batch = cluster_data[i: i+bsz]
                if batch:
                    yield batch
    else:
        fps = [smiles_to_fp(r[0].split('>')[0]) for r in data] # No pack
        fps = np.array(fps)
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, n_init=50, random_state=46)
            idx = kmeans.fit_predict(fps)

            print('avg bsz', len(data), '/', n_clusters, '=', len(data)/n_clusters)

            for bid in range(n_clusters):
                batch = []
                for i, d in enumerate(data):
                    if idx[i] == bid:
                        batch.append(d)
                        if len(batch) == bsz:
                            yield batch
                            batch = []
                if batch:
                    yield batch
        else:
            raise Exception("to add")


if __name__ == '__main__':
    success_count = 0
    result_list = []
    succ_thres = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
    succ_counter = {k: 0 for k in succ_thres}

    assert args.use_gnn_plan or args.use_value_fn
    
    with open(args.test_routes, 'rb') as route_f:
        routes = pickle.load(route_f)
        print('#routes', len(routes))
        i=0
        if args.n_proc == 1:
            gen = map
        else:
            pool = Pool(args.n_proc)
            gen = pool.imap
        max_succes_count_for_search = args.max_succes_count
        collect_data = (args.save_train_data_folder is not None) or (args.save_raw_data_folder is not None)
        gpu = args.gpu
        search_bsz = args.search_bsz
        batches = list(make_test_batch(routes, args.cluster_method, search_bsz, args.n_clusters))
        search_args = ((r, args.use_value_fn, max_succes_count_for_search, collect_data, args) for r in batches)

        total_iter = 0
        total_mol_nodes = 0
        total_rec_nodes = 0

        for search_data, result in tqdm(gen(search_worker, search_args), total=len(batches)):
            i += result['bsz']
            total_iter += result['iter']
            total_mol_nodes += result['n_mol_nodes']
            total_rec_nodes += result['n_rec_nodes']
            if result['success_count'] > 0:
                success_count += result['success_count']
                for k in succ_thres:
                    if result['avg_iter'] <= k:
                        succ_counter[k] += result['success_count']
                print(success_count, '/', i, '=', f'{success_count/i*100:.2f}')
        if args.n_proc > 1:
            pool.close()
    print(succ_counter)
    print({k:v/len(routes) for k, v in succ_counter.items()})
    print('Iter', total_iter, '/', len(batches), '=', total_iter / len(batches))
    print('#Rec', total_rec_nodes, '/', len(batches), '=', total_rec_nodes / len(batches))
    print('#Mol', total_mol_nodes, '/', len(batches), '=', total_mol_nodes / len(batches))
    