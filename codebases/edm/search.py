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
        "--data_path", type=str, default="./ablation_search",
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

    # data_num = len([d for d in os.listdir(os.path.join(args.data_path, args.split)) if ".pth" in str(d)])
    data_num = len([d for d in os.listdir(split_data_path) if ".pth" in str(d)])
    parameters = [
        "--config", "configs/edm-cifar10-32x32-uncond-vp.yml",
        "--sample_type", "unisampler",
        "--gpu", args.gpu,
        "--exp", f"exps/cifar10/eva6_test_{args.split}",
        "--sample",
        "--fid",
        "--statistics_dir", "dpm_solver_v3/edm-cifar10-32x32-uncond-vp/0.002_80.0_1200_1024",
        "--number_of_samples", "1000",
        "--dpmsolver_v3_t_start","80",
        "--dpmsolver_v3_t_end","0.002",
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
        while os.path.exists(os.path.join(split_data_path, f"{data_num}.pth")):
            print(f"{split_data_path}/{data_num}.pth exists")
            data_num += 1
        torch.save([new_decision, score], os.path.join(split_data_path,f"{data_num}.pth"))
        data_num += 1
        
        # update the search config
        with open(args.search_config, "r") as f:
            search_cfg = yaml.safe_load(f)
def restore_decision(path):
    import pathlib
    path = pathlib.Path(path)
    data_paths = [file for file in path.glob('**/*.{}'.format("pth"))]
    for data_path in data_paths:
        data = torch.load(data_path)
        decision = data[0]
        fid = data[1]
        decision["timesteps"] = decision["timesteps"].to("cpu")
        # if decision.get("use_afs",None) is not None:
        #     use_afs = decision["use_afs"]
        #     decision["afs"] = use_afs
        #     del decision["use_afs"]
        skip_coefficients = torch.linspace(1.0, 1.0, 15)
        decision["skip_coefficients"] = skip_coefficients
        torch.save([decision, fid], data_path)

if __name__ == "__main__":
    sys.exit(main())
    # fixed_noise = torch.randn(1024, 3, 32, 32)
    # torch.save(fixed_noise, "fixed_noise.pth")
    # id = os.getpid()
    # parameters = [
    #     "--config", "configs/edm-cifar10-32x32-uncond-vp.yml",
    #     "--sample_type", "unisampler",
    #     "--gpu", "0",
    #     "--exp", f"exps/cifar10/test_{id}",
    #     "--sample",
    #     "--fid",
    #     "--statistics_dir", "dpm_solver_v3/edm-cifar10-32x32-uncond-vp/0.002_80.0_1200_1024",
    #     "--number_of_samples", "1024",
    #     "--dpmsolver_v3_t_start","80",
    #     "--dpmsolver_v3_t_end","0.002",
    #     "--t_end", "0.002",
    #     "--t_start","80",
    # ]
    # runner = get_runner(parameters=parameters)
    # runner.generate_baseline_population()
    # import pathlib
    # base_path = "./search_data"
    # base_path = pathlib.Path(base_path)
    # data_paths = [file for file in base_path.glob('**/*.{}'.format("pth"))]
    # for data_path in data_paths:
    #     data = torch.load(data_path)
    #     decision = data[0]
    #     fid = data[1]
    #     decision["timesteps"] = decision["timesteps"].to("cpu")
    #     torch.save([decision, fid], data_path)
    # restore_decision("./search_data")