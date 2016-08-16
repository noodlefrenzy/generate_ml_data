import os
import argparse
import json
import numpy as np
import pandas as pd

def normal_dist(dist_conf):
    def curgen(cur_state):
        val = np.random.normal(dist_conf['mean'], dist_conf['sd'])
        val = max(dist_conf['min'], min(dist_conf['max'], val))
        if dist_conf['digits'] == 0:
            return str(int(val))
        else:
            return '{{0:0.{}f}}'.format(dist_conf['digits']).format(val)
    return curgen

def build_generator(gen_conf):
    if not 'dist' in gen_conf:
        def deferred_gen(cur_state):
            kvs = [('%s:%s' % (k, v)) for k, v in cur_state.items()]
            for kv in kvs:
                if kv in gen_conf:
                    print('Found kv ' + kv)
                    return build_generator(gen_conf[kv])(cur_state)
            print('Defaulting to default')
            return build_generator(gen_conf['default'])(cur_state)
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
            outf.write(id + '\t' + '\t'.join(curvals) + '\n')
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
            for (fname, fgen) in gens:
                curval = fgen(customers[id])
                curvals.append(curval)
            outf.write(id + '\t' + '\t'.join(curvals) + '\n')

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