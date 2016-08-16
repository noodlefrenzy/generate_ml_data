import os
import argparse
import json
import numpy as np
import pandas as pd

def build_generator(gen_conf):
    if gen_conf['dist'] == 'cumprob':
        keys = sorted(gen_conf['cats'].keys())
        vals = [gen_conf['cats'][k] for k in keys]
        cumprobs = list(zip(keys, np.cumsum(vals)))
        def curgen(cur_state):
            p = np.random.random()
            for (k, cump) in cumprobs:
                if p < cump:
                    return k
        return curgen
    elif gen_conf['dist'] == 'normal':
        def curgen(cur_state):
            val = np.random.normal(gen_conf['mean'], gen_conf['sd'])
            val = max(gen_conf['min'], min(gen_conf['max'], val))
            if gen_conf['digits'] == 0:
                return int(val)
            else:
                return (val, gen_conf['digits'])
        return curgen
    raise Exception('Invalid dist type: ' + gen_conf['dist'])

def gen_customers(config, num_customers, output_file):
    customers = {}
    with open(output_file, 'w', encoding='utf-8') as outf:
        features = config['Keys'] if 'Keys' in config else sorted(config.keys())
        outf.write('ID\t' + '\t'.join(features) + '\n')
        gens = [(k, build_generator(config[k])) for k in features]
        for idx in range(num_customers):
            id = str(idx)
            curvals = []
            customers[id] = {}
            for (fname, fgen) in gens:
                curval = fgen(customers[id])
                if type(curval) == tuple:
                    curval = '{{0:0.{}f}}'.format(curval[1]).format(curval[0])
                else:
                    curval = str(curval)
                curvals.append(curval)
                customers[id][fname] = curval
            outf.write(id + '\t' + '\t'.join(curvals) + '\n')
    return customers

def gen_products(config, customers, output_file):
    #with open(output_file, 'w', encoding='utf-8') as outf:
    pass

def gen_abandon(config, customers, output_file):
    #with open(output_file, 'w', encoding='utf-8') as outf:
    pass

def main():
    parser = argparse.ArgumentParser(description='Generate ML Data.')
    parser.add_argument('-n', '--ncust', type=int, required=True,
                        help='Number of customers to generate')
    parser.add_argument('-c', '--config', required=True, help='JSON file containing distributions for various features')
    parser.add_argument('-s', '--seed', type=int, default=1337, help='Seed value to allow deterministic random generation')
    args = parser.parse_args()
    np.random.seed(args.seed)
    config = json.load(open(args.config, 'r'))
    customers = gen_customers(config['customers'], args.ncust, './customers.tsv')
    gen_products(config['products'], customers, './products.tsv')
    gen_abandon(config['spend'], customers, './churn.tsv')

if __name__ == "__main__":
    main()