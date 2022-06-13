import dgl
import os
import pickle
import numpy as np
from scipy.stats import spearmanr, pearsonr
import math
import torch
import networkx as nx
from data_utils import make_dataset_ready_to_save, load_and_prep_data, make_dataset_from_saved_format


class BasePermuter():
    """The base permuter class that other permuters inherit from.
    This class contains general code that can be used in most experiments
    (i.e. mixing random, mode collapse, etc.)
    Parameters
    ----------
    evaluator : Evaluator
        The object that computes desired metrics
    helper : helper.ExperimentHelper
        General experiment helper --- logging results etc.
    Attributes
    ----------
    run_dir : str
        The directory to store the results
    saved_results : bool
        Description of attribute `saved_results`.
    evaluator : see parameters
    helper : see parameters
    """

    def __init__(self, evaluator, helper):
        """Initialize object and perform initial evaluation at step zero.
        Parameters
        ----------
        evaluator : Evaluator
            The object that computes desired metrics
        helper : helper.ExperimentHelper
            General experiment helper --- logging results etc.
        """
        self.featurized_permuted_set = None
        self.featurized_ref_set = None
        self.evaluator = evaluator
        self.helper = helper
        self.run_dir = helper.run_dir
        self.using_loaded_data = False
        self.saving = helper.args.save_perturb

        if helper.args.load_perturb:
            set1, set2 = load_and_prep_data(helper.args)
            if set1 is not None and set2 is not None:
                self.reference_set = set1
                self.permuted_sets = set2
                self.permuted_set = self.permuted_sets[0]
                self.using_loaded_data = True
                print('successfully loaded graphs from memory')
            else:
                print('loading pre-calculated data failed! reprocessing data!')

        if self.using_loaded_data:
            self.saving = False

        self.saved_results = False
        self.evaluate_graphs(iter=0)

    def perform_run(self):
        """Perform the desired experiment."""

        # TODO: works for perturbations we are working right now,
        # for the future make sure reference_set and permuted_set are
        # default name in permuters
        if not os.path.exists(self.helper.args.permute_data_drc):
            os.makedirs(self.helper.args.permute_data_drc)

        if self.saving:
            with open(self.helper.args.permute_data_drc + '/reference_set', 'wb+') as f:
                pickle.dump(make_dataset_ready_to_save(self.reference_set), f)

            with open(self.helper.args.permute_data_drc + f'/permuted_set_0', 'wb+') as f:
                pickle.dump(make_dataset_ready_to_save(self.permuted_set), f)

        if not self.using_loaded_data:
            iter = 1
            while 1:
                # Returns false if the experiment is complete
                success = self.permute_graphs(iter=iter)
                if not success:
                    break

                if self.saving:
                    # saving_permuted_sets.append(make_dataset_ready_to_save(self.permuted_set))
                    with open(self.helper.args.permute_data_drc + f'/permuted_set_{iter}', 'ab+') as f:
                        pickle.dump(make_dataset_ready_to_save(self.permuted_set), f)

                # Compute metrics
                self.evaluate_graphs(iter=iter)
                iter += 1
        else:
            for iter in range(1, len(self.permuted_sets)):
                self.permuted_set = self.permuted_sets[iter]
                # Compute metrics
                self.evaluate_graphs(iter=iter)

        # Save results, flip some metrics, compute correlations, etc.
        final_results = self.save_results_final()
        return final_results

    def permute_graphs(self, *args, **kwargs):
        """Abstract class."""
        raise Exception('Must be implemented by child class')

    def evaluate_graphs(self, iter):
        """Compute the desired metrics and save results.
        Parameters
        ----------
        iter : int
            The current iteration.
        """
        results = self(iter=iter)  # Compute metrics --- see __call__ methods
        self.__print_results(iter=iter, results=results)
        results = self.helper.save_results(results)  # Experiment logger saves results
        return results

    def save_results_final(self):
        """Save results, flip some metrics, compute correlations, etc."""
        if not self.saved_results:
            # Some metrics like P&R need to be "flipped" so they go in the
            # same direction as other metrics
            results = self.__flip_some_metrics()
            # Compute rank and pearson corrs.
            results = self.__get_correlations(results)

            # adding model loss to the saved data
            for res in results:
                res['model_loss'] = self.evaluator.model_loss

            print(results[0])  # Print corrs from experiment

            for key in results[0].keys():
                if key.startswith('rank_corr'):
                    print(key, ': ', results[0][key])

            self.helper.save_results(results)

            self.saved_results = True

            return results

    def __flip_some_metrics(self):
        results = self.helper.load_results()
        os.remove(os.path.join(self.helper.run_dir, 'results.h5'))  # Overwrite the results later

        keys_to_flip = ['precision', 'recall', 'density', 'coverage', 'f1_pr', 'f1_dc']
        tmp_dict = {key: [dct[key] for dct in results] for key in results[0].keys() for key_to_flip in keys_to_flip if
                    key_to_flip in key}
        for key in tmp_dict.keys():
            tmp_dict[key] = np.array(tmp_dict[key])
            if 'diff' not in key:
                tmp_dict[key] = max(tmp_dict[key]) - tmp_dict[key]
            elif 'diff' in key:
                tmp_dict[key] = -tmp_dict[key]
            else:
                raise Exception('unexpected key ' + key)

        for ix, res in enumerate(results):
            for key in tmp_dict.keys():
                res[key] = tmp_dict[key][ix]
        return results

    def __get_correlations(self, results):
        keys = list(results[0].keys())
        ratio_key = [key for key in keys if 'ratio' in key][0]
        ratios = [dct[ratio_key] for dct in results]

        for key in keys:
            if 'time' in key:
                continue
            metric = [dct[key] for dct in results]
            rank_corr, _ = spearmanr(ratios, metric)
            try:
                pearson_corr, _ = pearsonr(ratios, metric)
            except ValueError:
                pearson_corr = 0
            for dct in results:
                rank_corr = 0 if math.isnan(rank_corr) else rank_corr
                dct['rank_corr_' + key] = rank_corr
                pearson_corr = 0 if math.isnan(pearson_corr) else pearson_corr
                dct['pearson_corr_' + key] = pearson_corr

        return results

    def __print_results(self, iter, results):
        str = f'iter: {iter} '
        for key, val in results.items():
            str += f'{key}: {round(float(val), 4)} '
        self.helper.logger.info(str + '\n\n')

    def load_generated_graphs(self, args):
        """Load GRAN generated graphs for specified dataset.
        Parameters
        ----------
        args : Argparse dict
            Command line arguments parsed by argparse
        Returns
        -------
        list of DGL graphs
            List of graphs generated by GRAN trained on the specified datset.
        """
        dataset_name = args.dataset
        dataset_path = f'data/graphs/generations/gran/{dataset_name}.h5'
        generated_graphs = pickle.load(open(dataset_path, 'rb'))
        generated_graphs = [g.to(args.device) for g in generated_graphs]
        return generated_graphs

    def make_er_graphs(self, args, reference_set):
        """Generate ER graphs to resemble the given dataset.
        Parameters
        ----------
        args : Argparse dict
            Command line arguments parsed by argparse
        reference_set : list of DGL graphs
            The dataset used as reference for generating ER graphs.
        Returns
        -------
        list of DGL graphs
            List of DGL ER graphs that resemble given dataset. For ZINC,
            node/edge features are randomly selected according to their
            pmfs.
        """
        # Compute sparsity of each graph
        avg_p = [
            g.number_of_edges() / (g.number_of_nodes() ** 2)
            for g in reference_set]

        # Generate ER graphs
        generated_graphs = [
            nx.erdos_renyi_graph(
                g.number_of_nodes(), p, seed=np.random.choice(100))
            for g, p in zip(reference_set, avg_p)]

        generated_graphs = [dgl.DGLGraph(g) for g in generated_graphs]
        generated_graphs = [g.to(args.device) for g in generated_graphs]

        if args.dataset in ['zinc']:  # Randomly assign node/edge labels
            self.randomly_assign_labels(generated_graphs, reference_set, args.dataset)

        return generated_graphs

    def randomly_assign_labels(self, fake_graphs, reference_set, dataset):
        """Assign random node/edge labels to ER graphs.
        Parameters
        ----------
        fake_graphs : list of DGL graphs
            ER graphs generated by make_er_graphs that resemble specified dataset.
        reference_set : list of DGL graphs
            The dataset used as reference for generating ER graphs and
            determing node/edge label pmfs.
        Returns
        -------
        Nothing, but modifies fake_graphs in place to add attributes.
        """
        # Obtain pmf for node/edge features to sample from.
        # One-hot encoded so mean gives pmf.
        bg = dgl.batch(reference_set)  # Batch all graphs into single graph.
        node_pmf = bg.ndata['attr'].mean(axis=0).cpu().numpy()
        edge_pmf = bg.edata['attr'].mean(axis=0).cpu().numpy()

        num_node_attrs = node_pmf.shape[0]
        num_edge_attrs = edge_pmf.shape[0]

        for g in fake_graphs:
            # Sample node attrs. for each node in this graph
            fake_node_types = np.random.choice(
                num_node_attrs, p=node_pmf, size=g.number_of_nodes())
            # Convert to one-hot encoding
            one_hot = torch.eye(num_node_attrs).to(g.device)
            g.ndata['attr'] = one_hot[fake_node_types]  # Add features to graph

            # Sample edge attrs. for each node in this graph
            fake_edge_types = np.random.choice(
                num_edge_attrs, p=edge_pmf, size=g.number_of_edges())
            # Convert to one-hot encoding
            one_hot = torch.eye(num_edge_attrs).to(g.device)
            g.edata['attr'] = one_hot[fake_edge_types]  # Add features to graph
