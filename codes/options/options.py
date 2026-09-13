import os
import os.path as osp
import logging
import yaml
from utils.util import OrderedYaml

Loader, Dumper = OrderedYaml()


def parse(opt_path, gpu_ids=None, is_train=True):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    # export CUDA_VISIBLE_DEVICES
    if gpu_ids is not None: opt['gpu_ids'] = [int(x) for x in gpu_ids.split(',')]
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('exporting CUDA_VISIBLE_DEVICES=' + gpu_list)

    opt['is_train'] = is_train
    n_channels = opt['n_channels']
    noise_level = opt.get('noise_level', None)
    scale = opt['scale']

    if 'datasets' in opt:
        # datasets
        for phase, dataset in opt['datasets'].items():
            phase = phase.split('_')[0]
            print(dataset)
            dataset['phase'] = phase
            dataset['scale'] = scale
            if noise_level is not None:
                dataset['noise_level'] = noise_level
            is_lmdb = False
            if dataset.get('dataroot_GT', None) is not None:
                if isinstance(dataset['dataroot_GT'], list):
                    dataset['dataroot_GT'] = [osp.expanduser(dataroot_gt) for dataroot_gt in dataset['dataroot_GT']]
                else:
                    dataset['dataroot_GT'] = osp.expanduser(dataset['dataroot_GT'])
                    if dataset['dataroot_GT'].endswith('lmdb'):
                        is_lmdb = True
                # if dataset['dataroot_GT'].endswith('lmdb'):
                #     is_lmdb = True
            if dataset.get('dataroot_LQ', None) is not None:
                dataset['dataroot_LQ'] = osp.expanduser(dataset['dataroot_LQ'])
                if dataset['dataroot_LQ'].endswith('lmdb'):
                    is_lmdb = True
            dataset['data_type'] = 'lmdb' if is_lmdb else 'img'


    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))

    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_state'] = osp.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = osp.join(experiments_root, 'val_images')

        # change some options for debug mode
        if "debug" in opt["name"]:
            # opt['train']['val_freq'] = 8
            opt["val"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root


    # network
    opt['network_G']['n_channels'] = n_channels
    opt['network_G']['scale'] = scale

    if opt['network_G'].get('lf_flow', None) is not None:
        opt['network_G']['lf_flow']['nn_module'] = opt['network_G']['nn_module']
        for k, v in opt['network_G']['condFlow'].items():
            if opt['network_G']['lf_flow']['condFlow'].get(k, None) is None:
                opt['network_G']['lf_flow']['condFlow'][k] = v

    # for denoising
    if 'denoising' in opt['name'].lower():
        if 'soft' in opt['name'].lower():
            opt['network_G']['threshold_option'] = 's'
        elif 'hard' in opt['name'].lower():
            opt['network_G']['threshold_option'] = 'h'


    # relative learning rate
    if 'train' in opt:
        niter = opt['train']['niter']
        if 'T_period_rel' in opt['train']:
            opt['train']['T_period'] = [int(x * niter) for x in opt['train']['T_period_rel']]
        if 'restarts_rel' in opt['train']:
            opt['train']['restarts'] = [int(x * niter) for x in opt['train']['restarts_rel']]
        if 'lr_steps_rel' in opt['train']:
            opt['train']['lr_steps'] = [int(x * niter) for x in opt['train']['lr_steps_rel']]
        if 'lr_steps_inverse_rel' in opt['train']:
            opt['train']['lr_steps_inverse'] = [int(x * niter) for x in opt['train']['lr_steps_inverse_rel']]
        print(opt['train'])

    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter, network_label='G'):
    '''Check resume states and pretrain_model paths (overriding pretrain_paths)'''
    logger = logging.getLogger('base')
    model_label = 'pretrain_model_{}'.format(network_label)
    if opt['path']['resume_state']:
        if opt['path'].get(model_label, None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path'][model_label] = osp.join(opt['path']['models'], '{}_{}.pth'.format(resume_iter, network_label))
        logger.info('Set [pretrain_model_{}] to '.format(network_label) + opt['path'][model_label])
