# Defend
import os
import json
import argparse
from typing import Optional

import openbackdoor as ob
import pandas as pd
import torch
from openbackdoor.data import get_dataloader, wrap_dataset, load_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
from openbackdoor.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/base_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    # choose a victim classification model
    victim = load_victim(config["victim"])
    # choose attacker and initialize it with default parameters
    attacker = load_attacker(config["attacker"])
    defender = load_defender(config["defender"])
    # choose target and poison dataset
    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])
    print(poison_dataset)

    # target_dataset = attacker.poison(victim, target_dataset)
    # launch attacks
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    # backdoored_model = attacker.attack(victim, poison_dataset, config, defender)
    backdoored_model = torch.load('poison_model/bert_badnets_offenseval.pkl')
    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset, defender)
    display_results(config, results)

    # Fine-tune on clean dataset
    '''
    print("Fine-tune model on {}".format(config["target_dataset"]["name"]))
    CleanTrainer = ob.BaseTrainer(config["train"])
    backdoored_model = CleanTrainer.train(backdoored_model, wrap_dataset(target_dataset, config["train"]["batch_size"]))
    '''


if __name__ == '__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)

    main(config)
