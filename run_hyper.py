# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29
# @Author : Zihan Lin, Yupeng Hou
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn

import argparse
from ast import literal_eval
from numbers import Number

from recbole.quick_start import objective_function


METRIC_KEYS = ['recall@10', 'recall@20', 'recall@50', 'ndcg@10', 'ndcg@20', 'ndcg@50']


def _to_python_value(value):
    if isinstance(value, Number):
        return float(value)
    return value


def _format_metric_block(result_dict):
    lines = []
    for key in METRIC_KEYS:
        if key in result_dict:
            lines.append(f'{key}: {_to_python_value(result_dict[key])}')
    return '\n'.join(lines)


def _format_time_block(time_dict):
    return '\n'.join([
        f'training time: {_to_python_value(time_dict["train_time"]):.4f}s',
        f'valid time: {_to_python_value(time_dict["valid_time"]):.4f}s',
        f'test time: {_to_python_value(time_dict["test_time"]):.4f}s',
        f'total time: {_to_python_value(time_dict["total_time"]):.4f}s',
    ])


def _format_config_block(config_dict):
    return '\n'.join([
        f'T-Uni (tuni): {config_dict["tuni"]}',
        f'Epochs: {config_dict["epochs"]}',
        f'Learning Rate: {config_dict["learning_rate"]}',
        f'Gamma: {config_dict["gamma"]}',
        f'Weight Decay: {config_dict["weight_decay"]}',
        f'Encoder: {config_dict["encoder"]}',
        f'Train Batch Size: {config_dict["train_batch_size"]}',
    ])


def _format_result_section(title, result_dict):
    best_metric_name = result_dict.get('valid_metric', 'NDCG@20')
    lines = [title]
    lines.append(f'Test Result:\n{_format_metric_block(result_dict["test_result"])}')
    lines.append(f'Valid Result:\n{_format_metric_block(result_dict["best_valid_result"])}')
    lines.append(
        'Best Valid:\n'
        f'Best Epoch: {result_dict["best_epoch"]}\n'
        f'Best {best_metric_name}: {result_dict["best_valid_score"]}'
    )
    lines.append(f'Time (s):\n{_format_time_block(result_dict["time"])}')
    lines.append(f'Config:\n{_format_config_block(result_dict["config"])}')
    return '\n'.join(lines)


def _parse_fixed_config_dict(extra_args):
    fixed_config_dict = {}
    for arg in extra_args:
        if not arg.startswith('--'):
            continue
        key_value = arg[2:].split('=', 1)
        if len(key_value) != 2:
            continue
        key, value = key_value
        fixed_config_dict[key] = value
    return fixed_config_dict


def _load_choice_params(params_file):
    choice_params = {}
    with open(params_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            para_list = line.split(' ')
            if len(para_list) < 3:
                continue
            para_name, para_type, para_value = para_list[0], para_list[1], ''.join(para_list[2:])
            if para_type != 'choice':
                raise ValueError('Only `choice` params are supported for direct sweep mode.')
            choice_params[para_name] = literal_eval(para_value)
    return choice_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default=None, help='fixed config files')
    parser.add_argument('--params_file', type=str, default=None, help='parameters file')
    parser.add_argument('--output_file', type=str, default='hyper.result', help='output file')
    args, extra_args = parser.parse_known_args()
    fixed_config_dict = _parse_fixed_config_dict(extra_args)

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    if args.params_file:
        choice_params = _load_choice_params(args.params_file)
        if 'tuni' not in choice_params:
            raise ValueError('`tuni` must be provided in params_file for direct sweep mode.')

        best_result = None
        best_params = None
        all_results = []

        for tuni in choice_params['tuni']:
            current_config_dict = dict(fixed_config_dict)
            current_config_dict['tuni'] = tuni
            result = objective_function(
                config_dict=current_config_dict,
                config_file_list=config_file_list,
                saved=True,
            )
            all_results.append((tuni, result))
            print(_format_result_section(f'tuni={tuni}', result))
            print()

            score = result['best_valid_score']
            if best_result is None:
                best_result = result
                best_params = {'tuni': tuni}
            else:
                better = score > best_result['best_valid_score'] if result['valid_score_bigger'] else score < best_result['best_valid_score']
                if better:
                    best_result = result
                    best_params = {'tuni': tuni}

        print('all results:')
        for tuni, result in all_results:
            print(f'tuni={tuni}: best_valid_score={result["best_valid_score"]}, best_epoch={result["best_epoch"]}')
        print('best params: ', best_params)
        print(_format_result_section('best result', best_result))

        with open(args.output_file, 'w', encoding='utf-8') as fp:
            for tuni, result in all_results:
                fp.write(_format_result_section(f'tuni={tuni}', result))
                fp.write('\n\n')
            fp.write(f'Best Params: {best_params}\n\n')
            fp.write(_format_result_section('Best Result', best_result))
    else:
        result = objective_function(config_dict=fixed_config_dict, config_file_list=config_file_list, saved=True)
        print('best result: ')
        print(_format_result_section('best result', result))


if __name__ == '__main__':
    main()
