import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np  

from runners.diffusion import Diffusion
from utils.config_util import dict2namespace

torch.set_printoptions(sci_mode=False)

def get_logger(args):
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger()
    # create if none
    if not os.path.exists(args.exp):
        os.makedirs(args.exp)
    file_handler = logging.FileHandler(os.path.join(args.exp, "output.log"),mode='w')
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    logger.info("Conducting Command: %s", " ".join(sys.argv))
    return logger

def parse_args_and_config(parameters=None):
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument("--gpu", type=str, default="0", help="Specify the number of GPU device")
    parser.add_argument(
        "--doc",
        type=str,
        required=False,
        default="",
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "--sample_baseline",
        action="store_true",
        help="Whether evaluate several baseline method in a single run",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    # general
    parser.add_argument(
        "--return_intermediate", action="store_true"
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach ('generalized'(DDIM) or 'ddpm_noisy'(DDPM) or 'dpmsolver' or 'dpmsolver++' or 'unipc' or 'unisampler' or 'dpmsolver_v3')",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="logSNR",
        help="skip according to ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument(
        "--base_samples",
        type=str,
        default=None,
        help="base samples for upsampling, *.npz",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100, help="number of steps involved" # choose 100 to reproduce our results using DDIM (see Tab.2 in our paper https://arxiv.org/abs/2306.08860v1)
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--fixed_class", type=int, default=None, help="fixed class label for conditional sampling"
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--denoise", action="store_true", default=False)
    parser.add_argument("--lower_order_final", action="store_true", default=False)
    parser.add_argument("--thresholding", action="store_true", default=False)
    
    # for DPM-Solver
    parser.add_argument(
        "--dpm_solver_order", type=int, default=3, help="order of dpm-solver"
    )
    parser.add_argument(
        "--dpm_solver_atol", type=float, default=0.0078, help="atol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_rtol", type=float, default=0.05, help="rtol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_method",
        type=str,
        default="singlestep",
        help="method of dpm_solver ('adaptive' or 'singlestep' or 'multistep' or 'singlestep_fixed'",
    )
    parser.add_argument(
        "--dpm_solver_type",
        type=str,
        default="dpmsolver",
        help="type of dpm_solver ('dpmsolver' or 'taylor'",
    )
    
    # for Uni-PC
    parser.add_argument(
        "--uni_pc_order", type=int, default=3, help="order of uni-pc"
    )
    parser.add_argument(
        "--uni_pc_prediction_type", type=str, default="data_prediction", help="prediction type of uni-pc"
    )
    parser.add_argument(
        "--uni_pc_variant", type=str, default='bh1', help="B(h) of uni-pc"
    )
    parser.add_argument(
        "--uni_pc_disable_corrector", action="store_true",
    )
    
    # for Uni-Sampler
    parser.add_argument(
        "--uni_sampler_decision_type", type=str, default=None, help="type of decision for uni-sampler, can be chosen from ['dpmsolver', 'unipc' , 'from_search_space']"
    )
    parser.add_argument(
        "--uni_sampler_method", type=str, default="multistep", help="singlestep or multistep, only works if the decision is not got in advance"
    )
    parser.add_argument(
        "--uni_sampler_order", type=int, default=3, help="the taylor order of uni-sampler, only works if the decision is not got in advance"
    )
    parser.add_argument(
        "--t_start", type=float, default=1.0, help="t_T"
    )
    parser.add_argument(
        "--t_end", type=float, default=0.001, help="t_0"
    )
    parser.add_argument(
        "--load_decision", type=str, default=None, help="load a decision"
    )
    
    # for dataset generating
    parser.add_argument(
        "--decision_config", type=str, default="./uni_sampler/configs/decision_config_v2_v5.yml", help="configuration for decision sampling"
    )
    parser.add_argument(
        "--search_space_version", type=str, default=None, help="version of the search space",
    )
    parser.add_argument(
        "--population_generate", type=str, default=None, help="load a population for dataset generation"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None
    )
    parser.add_argument(
        "--split", type=str, default=None
    )
    parser.add_argument(
        "--data_num", type=int, default=700
    )
    
    # reduced fid evalaution
    parser.add_argument(
        "--reduced_ratio", type=str, default=None
    )
    
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--port", type=str, default="12355")
    
    # store statistics dir in configs, don't need to pass it as an argument
    parser.add_argument(
        "--statistics_dir", type=str, default=None, help="not used now, please store it in configs."
    )
    parser.add_argument("--dpmsolver_v3_order", type=int, default=3, help="Order of DPM Solver v3.")
    parser.add_argument("--p_pseudo", action="store_true", help="Use P-pseudo if set.")
    parser.add_argument("--use_corrector", action="store_true", help="Use corrector if set.")
    parser.add_argument("--c_pseudo", action="store_true", help="Use C-pseudo if set.")
    parser.add_argument("--degenerated", action="store_true", help="Use degenerated mode if set.")
    parser.add_argument("--dpmsolver_v3_t_start", type=float, default=1, help="Start time for DPM Solver v3 EMS.")
    parser.add_argument("--dpmsolver_v3_t_end", type=float, default=1e-4, help="End time for DPM Solver v3 EMS.")

    parser.add_argument("--number_of_samples", type=int, default=None, help="Number of samples to generate.")
    parser.add_argument("--fid_result_path", type=str, default=None, help="Path to the FID result file.")
    
    parser.add_argument("--afs", type=str, default="no_afs", help="AFS method to use. support [no_afs, zero_x0,average_x0]")
    
    if parameters:
        args = parser.parse_args(parameters)
    else:
        args = parser.parse_args()
    
    # for dataset generating
    if args.dataset_path is not None:
        args.exp = os.path.join(args.dataset_path, args.split)
        
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    config_path = args.config if os.path.isfile(args.config) else os.path.join("configs", args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if not args.test and not args.sample and not args.dataset_generate and not args.sample_baseline and not args.population_evaluate:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni or args.dataset_generate:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    os.makedirs(args.log_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample or args.dataset_generate or args.sample_baseline or args.population_evaluate:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples"
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)
                        
    logger = get_logger(args)

    # add device
    if args.gpu and torch.cuda.is_available():
        args.gpu_flag = True
        device = torch.device('cuda')
        gpus = [int(d) for d in args.gpu.split(',')]
        args.gpu = gpus
        torch.cuda.set_device(gpus[0]) # currently only training & inference on single card is supported.
        logger.info("Using GPU(s). Available gpu count: {}".format(torch.cuda.device_count()))
    else:
        device = torch.device('cpu')
        logger.info("Using cpu!")
    new_config.device = device

    torch.backends.cudnn.benchmark = True

    return args, new_config, logger

def get_runner(parameters=None):
    args, config, logger = parse_args_and_config(parameters)
    return Diffusion(args, config, logger)

def main():
    args, config, logger = parse_args_and_config()
    logger.info("Writing log file to {}".format(args.log_path))
    logger.info("Exp instance id = {}".format(os.getpid()))
    logger.info("Exp comment = {}".format(args.comment))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs("temp", exist_ok=True)
    try:
        runner = Diffusion(args, config, logger)
        if args.sample:
            runner.sample()
        elif args.sample_baseline:
            runner.sample_different_sampler_nfe()
        elif args.test:
            runner.test()
        else:
            runner.train()
    except Exception:
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    sys.exit(main())
