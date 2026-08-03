import yaml, numpy as np, sys, os
sys.path.insert(0, '.')
from dataset.dataset_test import MolTestDatasetWrapper
import finetune

TASKS = {
    'BBBP':    {'data': 'data/bbbp/BBBP.csv',              'target': 'p_np',        'task': 'classification'},
    'BACE':    {'data': 'data/bace/bace.csv',              'target': 'Class',       'task': 'classification'},
    'HIV':     {'data': 'data/hiv/HIV.csv',                'target': 'HIV_active',  'task': 'classification'},
    'ClinTox': {'data': 'data/clintox/clintox.csv',        'target': 'CT_TOX',      'task': 'classification'},
    'Tox21':   {'data': 'data/tox21/tox21.csv',            'target': 'NR-AR',       'task': 'classification'},
    'SIDER':   {'data': 'data/sider/sider.csv',            'target': 'Hepatobiliary disorders', 'task': 'classification'},
    'MUV':     {'data': 'data/muv/muv.csv',                'target': 'MUV-692',     'task': 'classification'},
    'FreeSolv':{'data': 'data/freesolv/freesolv.csv',      'target': 'expt',        'task': 'regression'},
    'ESOL':    {'data': 'data/esol/esol.csv',              'target': 'measured log solubility in mols per litre', 'task': 'regression'},
    'Lipo':    {'data': 'data/lipophilicity/Lipophilicity.csv', 'target': 'exp',     'task': 'regression'},
    'qm7':     {'data': 'data/qm7/qm7.csv',                'target': 'u0_atom',     'task': 'regression'},
    'qm8':     {'data': 'data/qm8/qm8.csv',                'target': 'E1-CC2',      'task': 'regression'},
    'qm9':     {'data': 'data/qm9/qm9.csv',                'target': 'mu',          'task': 'regression'},
}

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gin'  # gin or gcn

for task_name, cfg in TASKS.items():
    # Load base config
    config = yaml.load(open('config_finetune_molnet.yaml'), Loader=yaml.FullLoader)
    config['task_name'] = task_name
    config['model_type'] = MODEL
    config['fine_tune_from'] = f'pretrained_{MODEL}'
    config['dataset']['data_path'] = cfg['data']
    config['dataset']['target'] = cfg['target']
    config['dataset']['task'] = cfg['task']

    if task_name == 'ClinTox':
        config['dataset']['target'] = 'FDA_APPROVED'
        config['epochs'] = 50

    results = []
    for i in range(3):
        print(f'{task_name} run {i+1}/3...', end=' ', flush=True)
        dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
        ft = finetune.FineTune(dataset, config)
        ft.train()
        if cfg['task'] == 'classification':
            results.append(ft.roc_auc)
            print(f'AUC={ft.roc_auc:.4f}')
        else:
            if task_name in ['qm7', 'qm8', 'qm9']:
                results.append(ft.mae)
                print(f'MAE={ft.mae:.4f}')
            else:
                results.append(ft.rmse)
                print(f'RMSE={ft.rmse:.4f}')

    r = np.array(results)
    print(f'  {task_name} {MODEL}: {r.mean():.4f} +- {r.std():.4f}\n')

    # Save
    os.makedirs(f'molnet_results/{MODEL}', exist_ok=True)
    with open(f'molnet_results/{MODEL}/{task_name}.txt', 'w') as f:
        f.write(f'{r.mean():.4f},{r.std():.4f}\n')

# Summary
print(f'\n{"="*60}')
print(f'MoleculeNet Results: {MODEL}')
print(f'{"="*60}')
for task_name, cfg in TASKS.items():
    fpath = f'molnet_results/{MODEL}/{task_name}.txt'
    if os.path.exists(fpath):
        val = open(fpath).read().strip()
        metric = 'RMSE' if cfg['task'] == 'regression' and task_name not in ['qm7','qm8','qm9'] else ('MAE' if task_name in ['qm7','qm8','qm9'] else 'AUC')
        print(f'  {task_name:12s}: {val} ({metric})')
