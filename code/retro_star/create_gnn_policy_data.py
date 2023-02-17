import os
import torch
import pickle
from retro_star.alg.retro_graph import MolNode, RecNode, SearchGraph, smiles_to_fp
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm

from multiprocessing import Pool
from retro_star.common import args, prepare_starting_molecules, prepare_mlp
torch.multiprocessing.set_sharing_strategy('file_system')

def process(route, known_mols, expand_fn): # plan is a list of string split by >>
    ret = []
    if route is None:
        return ret

    plan = route.serialize().split('|')
    search_graph = SearchGraph([plan[0].split('>')[0]], known_mols)
    all_mol_in_path = set()
    for reaction in plan:
        prod = reaction.split('>')[0]
        all_mol_in_path.add(prod)

    for reaction in plan:
        # expand the graph
        prod, _, real_reactants = reaction.split('>')
        real_reactants = real_reactants.split('.')

        if prod not in search_graph.smiles_to_node:
            continue
        prod_node = search_graph.smiles_to_node[prod]
        result = expand_fn.run(prod, topk=50)
        if result is None or not result['scores']:
            continue
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
        
        new_success_mols, new_mols = search_graph.expand(prod_node, reactant_lists, templates, costs)
        for new_mol in new_success_mols:
            search_graph.push_up_success(new_mol)
        prod_node.open = False

        # collect data
        node_to_idx = {}

        for mol in search_graph.smiles_to_node.values():
            node_to_idx[mol]=len(node_to_idx)
            for rec in mol.children:
                node_to_idx[rec] = len(node_to_idx)
        n_node = len(node_to_idx) 
        x = [None] * n_node
        x_feature = [None] * n_node
        edge_index = [[], []]
        edge_attr = []
        y = [None] * n_node
        
        for mol in search_graph.smiles_to_node.values():
            node_idx = node_to_idx[mol] 
            x[node_idx] = smiles_to_fp(mol.smiles, pack=True)
            # 0: type (mol 0 rec 1)
            # 1: success (false 0 true 1)
            # 2: is target (false 0 true 1)
            # 3: reaction cost (0 for mol)
            # 4: history cost
            x_feature[node_idx] = [
                0,
                1 if mol.success else 0,
                1 if mol in search_graph.target_mols else 0,
                0,
                mol.cost
            ]

            if mol.open:
                y[node_idx] = 1 if mol.smiles in all_mol_in_path else 0
            else:
                y[node_idx] = -1

            for rec in mol.children:
                rec_node_idx = node_to_idx[rec]
                x[rec_node_idx] = np.packbits(np.zeros(2048, dtype=bool))
                x_feature[rec_node_idx] = [
                    1,
                    1 if mol.success else 0,
                    0,
                    rec.reaction_cost,
                    rec.cost
                ]
                y[rec_node_idx] = -1

                # add edges
                edge_index[0].extend([node_idx, rec_node_idx])
                edge_index[1].extend([rec_node_idx, node_idx])
                edge_attr.extend([0, 1]) # 0 s->T, 1 t->s

                for grand in rec.children:
                    grand_idx = node_to_idx[grand]
                    edge_index[0].extend([rec_node_idx, grand_idx])
                    edge_index[1].extend([grand_idx, rec_node_idx])
                    edge_attr.extend([0, 1]) # 0 s->T, 1 t->s

        # assert None not in x
        # assert None not in x_feature
        # assert None not in y
        assert len(edge_index[0]) == len(edge_index[1]) == len(edge_attr)

        if max(y) == 1:
            ret.append(
                Data(
                    x = torch.Tensor(x),
                    x_feature = torch.Tensor(x_feature),
                    y = torch.Tensor(y),
                    edge_index = torch.Tensor(edge_index).long(),
                    edge_attr = torch.Tensor(edge_attr)
                )
            )

    return ret

def worker(func_args):
    route, args = func_args
    if not hasattr(worker, 'starting_mols'):
        worker.starting_mols = prepare_starting_molecules(args.starting_molecules)

    if not hasattr(worker, 'one_step'):
        worker.one_step = prepare_mlp(args.mlp_templates, args.mlp_model_dump, gpu=args.gpu)

    return process(route, worker.starting_mols, worker.one_step)


if __name__ == '__main__':
    with open(args.test_routes, 'rb') as route_f:
        routes = pickle.load(route_f)['routes']
        result = []
        func_args = ((route, args) for route in routes)
        if args.n_proc == 1:
            gen = map(worker, func_args)
        else:
            if args.gpu == -1:
                pool = Pool(args.n_proc)
            else:
                pool = torch.multiprocessing.get_context("spawn").Pool(args.n_proc, maxtasksperchild=1)
            gen = pool.imap(worker, func_args)
        
        for graphs in tqdm(gen, total=len(routes)):
            result.extend(graphs)
        
        if args.n_proc > 1:
            pool.close()
        
        print('total graph', len(result))
        torch.save(result, os.path.join(args.save_train_data_folder, 'graph.pt'))

