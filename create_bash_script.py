import numpy as np
np.random.seed(0)


num_layers_lst = [3]
hidden_dims_lst = [16, 32]
# seeds = list(range(5))
seeds = [0]
permutation_exps = ['mixing-random', 'mode-collapse', 'mode-dropping', 'rewiring-edges']
# permutation_exps = ['rewiring-edges']
datasets = ['lobster', 'grid', 'ego', 'community', 'proteins', 'zinc']
# datasets = ['lobster', 'grid',  'zinc', 'ego', 'community', 'proteins']
# gnns = ['graphcl', 'infograph', 'gin-random']
gnns = ['graphcl']

deg_feats = True
clus_feats = True
results_directory = 'testing'
if clus_feats:
    results_directory += '_clus_feats'
elif deg_feats:
    results_directory += '_deg_feats'
else:
    results_directory += '_no_feats'
results_directory += '_in_progress'

save_perturb = False
load_perturb = False

use_predfined_hyperparams = False
is_parallel = False




def create_commands():
    bash_cmds = ['#!/bin/bash']

    def generate_gnn_commands():
        commands = []
        for exp in permutation_exps:
            for dataset in datasets:
                for gnn in gnns:
                    for num_layers in num_layers_lst:
                        for hidden_dims in hidden_dims_lst:
                            for seed in seeds:
                                command = f'python main.py --seed={seed} --permutation_type={exp} --dataset={dataset}' \
                                          f' --num_layers={num_layers} --hidden_dim={hidden_dims}' \
                                          f' --feature_extractor={gnn} --results_directory={results_directory}'
                                if deg_feats:
                                    command += ' --deg_feats=True'
                                if clus_feats:
                                    command += ' --clus_feats=True --orbit_feats=True'
                                if save_perturb:
                                    command += ' --save_perturb=True'
                                if load_perturb:
                                    command += ' --load_perturb=True'
                                if use_predfined_hyperparams:
                                    command += ' --use_predefined_hyperparams=True'
                                if is_parallel:
                                    command += ' --is_parallel=True'
                                commands += [command]
        return commands

    bash_cmds += generate_gnn_commands()
    return bash_cmds


bash_cmds = create_commands()
open('all_commands.sh', 'w').write('\n'.join(bash_cmds))


if __name__ == '__main__':
    create_commands()