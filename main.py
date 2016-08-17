import os
import argparse
import json
import numpy as np
import pandas as pd

def normal_dist(dist_conf):
    def curgen(cur_state):
        mean = dist_conf['mean']
        if type(mean) == str:
            mean = cur_state[mean]
        val = np.random.normal(mean, dist_conf['sd'])
        val = max(dist_conf['min'], min(dist_conf['max'], val))
        if dist_conf['digits'] == 0:
            return int(val)
        else:
            return val
    return curgen

ops = {
    'eq': lambda key, val: (lambda state: state[key] == val),
    'lt': lambda key, val: (lambda state: state[key] < val),
    'le': lambda key, val: (lambda state: state[key] <= val),
    'gt': lambda key, val: (lambda state: state[key] > val),
    'ge': lambda key, val: (lambda state: state[key] >= val),
    'ne': lambda key, val: (lambda state: state[key] != val)
}

def build_generator(gen_conf):
    if 'constrained' in gen_conf:
        constrained = []
        for constraint in gen_conf['constrained']:
            tests = [ops[x['op']](x['key'], x['val']) for x in constraint['rules']]
            constrained.append({
                'name': constraint['name'],
                'tests': tests,
                'dist': build_generator(constraint)
            })
        default_dist = build_generator(gen_conf['default'])
        def deferred_gen(cur_state):
            for constraint in constrained:
                if all([test(cur_state) for test in constraint['tests']]):
                    print('Rule "%s" matched' % constraint['name'])
                    return constraint['dist'](cur_state)
            #print('Defaulting to default')
            return default_dist(cur_state)
        return deferred_gen
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
        return normal_dist(gen_conf)
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
                curvals.append(curval)
                customers[id][fname] = curval
            outf.write(id + '\t' + '\t'.join([str(x) for x in curvals]) + '\n')
    return customers

def gen_products(config, customers, output_file):
    with open(output_file, 'w', encoding='utf-8') as outf:
        products = config['names']
        rating_gen = normal_dist(config['ratings_dist'])
        outf.write('customer_id\tproduct\trating\n')
        for id, cust in customers.items():
            for product in products:
                rating = int(rating_gen(cust))
                outf.write('%s\t%s\t%d\n' % (id, product, rating))

def gen_abandon(config, customers, output_file):
    with open(output_file, 'w', encoding='utf-8') as outf:
        features = config['Keys'] if 'Keys' in config else sorted(config.keys())
        outf.write('ID\t' + '\t'.join(features) + '\n')
        gens = [(k, build_generator(config[k])) for k in features]
        for id, cust in customers.items():
            curvals = []
            curstate = customers[id].copy()
            for (fname, fgen) in gens:
                curval = fgen(curstate)
                curstate[fname] = curval
                curvals.append(curval)
            outf.write(id + '\t' + '\t'.join([str(x) for x in curvals]) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate ML Data.')
    parser.add_argument('-n', '--ncust', type=int, required=True,
                        help='Number of customers to generate')
    parser.add_argument('-c', '--config', required=True, help='JSON file containing distributions for various features')
    parser.add_argument('--seed', type=int, default=1337, help='Seed value to allow deterministic random generation')
    parser.add_argument('-s', '--suffix', type=str, default='', help='Suffix for output files')
    args = parser.parse_args()
    np.random.seed(args.seed)
    config = json.load(open(args.config, 'r'))
    suffix = '' if args.suffix == '' else ('_' + args.suffix)
    customers = gen_customers(config['customers'], args.ncust, './customers%s.tsv' % suffix)
    gen_products(config['products'], customers, './products%s.tsv' % suffix)
    gen_abandon(config['spend'], customers, './churn%s.tsv' % suffix)

if __name__ == "__main__":
    main()