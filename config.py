import argparse


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Experiment args')

        self.parser.add_argument(
            '--no_cuda', action='store_true',
            help='Flag to disable cuda')

        self.parser.add_argument(
            '--seed', default=42, type=int,
            help='the random seed to use')

        self.parser.add_argument(
            '--dataset', default='lobster',
            choices=['grid', 'lobster', 'lego', 'proteins',
                     'community', 'ego', 'zinc'],
            help='The dataset to use')

        self.parser.add_argument(
            '--permutation_type', default='mode-collapse',
            choices=['sample-size-random', 'mixing-gen', 'mixing-random',
                     'rewiring-edges', 'mode-collapse', 'mode-dropping',
                     'computation-eff-qty', 'computation-eff-size',
                     'computation-eff-edges', 'randomize-nodes',
                     'randomize-edges'],
            help='The permutation (experiment) to run')

        self.parser.add_argument(
            '--step_size', default=0.01, type=float,
            help='Many experiments have a "step size", e.g. in mixing random\
                graphs, the step size is the percentage (fraction) of random\
                graphs added at each time step.')

        self.parser.add_argument(
            '--results_directory', type=str, default='testing',
            help='Results are saved in experiment_results/{results_directory}')

        self.parser.add_argument(
            '--save_perturb', default=False, type=bool,
            help='If true it will save the graphs from the result of the perturbation.'
        )

        self.parser.add_argument(
            '--load_perturb', default=False, type=bool,
            help='If true it will load the previously perturbed data instead of generating it (if available).'
        )

        self.parser.add_argument(
            '--is_parallel', default=False, type=bool,
            help="For degree, clustering, orbits, and spectral MMD metrics. Or node feature extractor in GIN.\
            Whether to compute graph statistics in parallel or not.")

        self.parser.add_argument(
            '--max_workers', default=4, type=int,
            help="If is_parallel is true, this sets the maximum number of\
                    workers.")

        self.parser.add_argument(
            '--use_predefined_hyperparams', default=False, type=bool,
            help="If it is true, it uses hyperparameters predefined from before for each dataset.\
                 This ignores the hyperparameters in the config/args"
        )

        self.parser.add_argument(
            '--split', default=False, type=bool,
            help="Split the data between the train and test?")

        self.parser.add_argument(
            '--split_ratio', default=0.5, type=float,
            help="Ratio to split dataset between train and test data.")

        # gin_parser = subparsers.add_parser('gnn')
        self.parser.add_argument(
            '--feature_extractor', default='graphcl',
            choices=['graphcl', 'gin-random', 'infograph'],
            help='The GNN to use')

        self.parser.add_argument(
            '--num_layers', default=3, type=int,
            help='The number of prop. rounds in the GNN')

        self.parser.add_argument(
            '--hidden_dim', default=32, type=int,
            help='The node embedding dimensionality. Final graph embed size \
            is hidden_dim * (num_layers - 1)')

        self.parser.add_argument(
            '--epochs', default=100, type=int,
            help='number of epochs for training')

        self.parser.add_argument(
            '--init', default='orthogonal', type=str,
            choices=['default', 'orthogonal'],
            help="The weight init. method for the GNN. Default is PyTorchs\
            default init.")

        self.parser.add_argument(
            '--retrain', default=True, type=bool,
            help='Yes: retrain the gin network/ No: use a pretrained one if available')

        self.parser.add_argument(
            '--model_name', default='temp', type=str,
            help='name of the model to save')

        self.parser.add_argument(
            '--limit_lip', default=True, type=bool,
            help='Should model limit the lipchitzness factor of the layers?')

        self.parser.add_argument(
            '--lip_factor', default=1.0, type=float,
            help='lipchitzness factor of the mlp layers in GConv'
        )

        self.parser.add_argument(
            '--const_feats', default=True, type=bool,
            help='Should dataset add constant features?')

        self.parser.add_argument(
            '--deg_feats', default=True, type=bool,
            help='Should dataset add target normalized degree features?')

        self.parser.add_argument(
            '--clus_feats', default=False, type=bool,
            help='Should dataset add clustering features?')

        self.parser.add_argument(
            '--orbit_feats', default=False, type=bool,
            help='Should dataset add orbit features?')

    def parse(self):
        """Parse the given command line arguments.
        Parses the command line arguments and overwrites
        some values to ensure compatibility.
        Returns
        -------
        Argparse dict: The parsed CL arguments
        """
        args = self.parser.parse_args()
        args.results_directory = '' if args.results_directory is None \
            else args.results_directory + '/'

        # args.use_degree_features = True

        if args.dataset == 'zinc':
            args.input_dim = 28  # The number of node features in zinc
            args.edge_feat_dim = 4  # Num edge feats. in zinc
        else:
            args.input_dim = 1  # We use node degree as an int. as node feats.
            args.edge_feat_dim = None  # No edge features for non-zinc datasets

        args.results_directory += \
            f'{args.feature_extractor}/{args.permutation_type}/{args.dataset}'
        args.graph_embed_size = (args.num_layers - 1) * args.hidden_dim

        if args.use_predefined_hyperparams:
            if args.dataset == 'lobster':
                args.hidden_dim = 16
                args.num_layers = 2
                args.epochs = 40
            elif args.dataset == 'grid':
                args.hidden_dim = 16
                args.num_layers = 3
                args.epochs = 20
            else:
                args.hidden_dim = 32
                args.num_layers = 3
                args.epochs = 100

        if args.model_name == 'temp':
            args.model_name = f'{args.feature_extractor}_{args.dataset}_{args.num_layers}_{args.hidden_dim}_def_feat_{args.deg_feats}_cluster_feat_{args.clus_feats}' \
                              f'_{args.orbit_feats}_{args.epochs}_{args.limit_lip}_{args.lip_factor}_{args.split_ratio}'

        args.permute_data_drc = f'data/perturb_data/{args.permutation_type}/{args.dataset}/{args.seed}'

        return args
