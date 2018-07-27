import os
from argparse import ArgumentParser

import yaml

from postprocessing.saliency_maps.visualize_tcga import visualize_tcga
from postprocessing.saliency_maps.visualize_tupac import visualize_tupac
from trainers import tupac, tcga


def init_arg_parser():
    parser = ArgumentParser()
    parser.add_argument('-y', '--yaml', dest="yaml", help="Yaml config file")
    parser.add_argument('codename', choices=['tupac', 'tcga', 'tupac_sm', 'tcga_sm'])
    args = parser.parse_args()

    return args


def add_yaml_tags():
    # define custom tag handler
    def join_path(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.join(*seq)

    # register the tag handler
    yaml.add_constructor('!join_path', join_path)
    return yaml


if __name__ == '__main__':
    args = init_arg_parser()
    yaml = add_yaml_tags()
    with open(args.yaml, 'r') as cfgfile:
        config = yaml.load(cfgfile)

    if args.codename == "tupac":
        tupac.single_run(config)
    elif args.codename == "tcga":
        tcga.single_run(config)
    elif args.codename == "tupac_sm":
        visualize_tupac(config)
    elif args.codename == "tcga_sm":
        visualize_tcga(config)
