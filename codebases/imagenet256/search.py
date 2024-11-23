import torch
import numpy as np
import random
import yaml
import argparse
import os
import sys
from utils.search_utils import get_population, select_parents, crossover, mutate
from main import get_runner
from runners.diffusion import Diffusion

def eval_decisions(runner:Diffusion, decisions:dict):
    fid = runner.sample(number_of_samples=1000, decisions=decisions)
    return fid

def main():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--search_config", type=str, default=f"./configs/evo_search.yml", help="path of search config"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
    )
    parser.add_argument(
        "--data_path", type=str, default="./search_data",
    )
    parser.add_argument(
        "--split", type=str, default=None,
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
    )
    args = parser.parse_args()
    
    # set seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        # use current time as seed
        import time
        seed = int(time.time())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # set data save path
    if args.split is None:
        args.split = args.gpu
    split_data_path = os.path.join(args.data_path, args.split)
    os.makedirs(split_data_path, exist_ok=True)
    
    # search
    with open(args.search_config, "r") as f:
        search_cfg = yaml.safe_load(f)

    data_num = len([d for d in os.listdir(split_data_path) if ".pth" in str(d)])
    # data_num = len([d for d in os.listdir(args.data_path) if ".pth" in str(d)])
    parameters = [
        "--config", "configs/imagenet256_guided.yml",
        "--sample_type", "unisampler",
        "--gpu", args.gpu,
        "--exp", f"exps/imagenet256/split_{args.split}",
        "--sample",
        "--fid",
        "--statistics_dir", "dpm_solver_v3/imagenet256_guided/500_1024",
        "--number_of_samples", "1000",
    ]
    runner = get_runner(parameters=parameters)
    while(1):
        
        population = get_population(args.data_path, search_cfg)
        
        if random.uniform(0, 1) < search_cfg["crossover"]["prob"]:
            count = 0
            while True:
                parents_1, parents_2 = select_parents(population, search_cfg, num=2)
                count += 1
                if len(parents_1[0]["orders"]) == len(parents_2[0]["orders"]) or count > 5:
                    break
            new_decision = crossover(parents_1[0], parents_2[0],config=search_cfg)
        else:
            new_decision = None
        
        if new_decision is not None:
            parent = new_decision
        else:
            parent = select_parents(population, search_cfg, num=1)[0]
        new_decision = mutate(parent, search_cfg)
        
        score = eval_decisions(runner, new_decision)
        print(f"{data_num}:{score}")
        while os.path.exists(os.path.join(split_data_path, f"{data_num}.pth")):
            print(f"{split_data_path}/{data_num}.pth exists")
            data_num += 1
        torch.save([new_decision, score], os.path.join(split_data_path, f"{data_num}.pth"))
        data_num += 1
        
        # update the search config
        with open(args.search_config, "r") as f:
            search_cfg = yaml.safe_load(f)
        
if __name__ == "__main__":
    sys.exit(main())