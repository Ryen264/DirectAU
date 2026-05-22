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

from recbole.quick_start import objective_function


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
            print(f'tuni={tuni}')
            print('best valid result:')
            print(result['best_valid_result'])
            print('test result:')
            print(result['test_result'])
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
            print(f'tuni={tuni}: best_valid_score={result["best_valid_score"]}, best_valid_result={result["best_valid_result"]}, test_result={result["test_result"]}')
        print('best params: ', best_params)
        print('best result: ')
        print(best_result)

        with open(args.output_file, 'w', encoding='utf-8') as fp:
            for tuni, result in all_results:
                fp.write(f'tuni={tuni}\n')
                fp.write(f'best_valid_score: {result["best_valid_score"]}\n')
                fp.write(f'best_valid_result: {result["best_valid_result"]}\n')
                fp.write(f'test_result: {result["test_result"]}\n\n')
            fp.write(f'best params: {best_params}\n')
            fp.write(f'best result: {best_result}\n')
    else:
        result = objective_function(config_dict=fixed_config_dict, config_file_list=config_file_list, saved=True)
        print('best result: ')
        print(result)


if __name__ == '__main__':
    main()
