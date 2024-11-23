# Unified Sampling Framework (USF)

This is the an simple and unorganized repository of the paper "A Unified Sampling Framework for Solver Searching of Diffusion Probabilistic Models". We provide some example solver schedules to reproduce our results in the paper.

## Dataset, Checkpoint and FID Stats
Currently we only support CIFAR-10 with ScoreSDE model. Please check the config files in `configs/`.

Please put the download model to `checkpoints/`. For model loading, please check this code for details: `functions/ckpt_util.py`. If you want to use other FID Stats to further verify our results, please put it in the `activations/` and adjust its data type to a dict with keys "m" and "s"(see our example in `activations/cifar10/statistic.pth` and our code in `evaluate/fid_score.py`). 

| Config File            | Checkpoint                                                   | Other FID Stats                                              |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| cifar10_continouous.yml            | Download [ScoreSDE checkpoint on CIFAR-10](https://drive.google.com/drive/folders/1ZMLBiu9j7-rpdTQu8M2LlHAEQq4xRYrj) in `checkpoints/`. | [Download](https://drive.google.com/drive/folders/1_OpTXVPLffZM8BG-V3Ahsxk99aqxW7C3?usp=sharing) `fid_stats_cifar10_train_pytorch.npz` |

## Command
We provide a command for sampling and FID evaluation here.
```
python main.py --config cifar10_continuous.yml --sample_type unisampler --gpu 5 --exp exps/cifar10/nfe4 --sample --fid --load_decision ./solver_schedules/cifar10_4nfe.pth

python main.py --config cifar10_continuous.yml --sample_type dpmsolver_v3 --gpu 4 --exp exps/cifar10/dpm_solver_v3 --sample --fid --statistics_dir dpm_solver_v3/cifar10_ddpmpp_deep_continuous/0.0001_1200_4096 --lower_order_final --use_corrector --timesteps 8 --dpmsolver_v3_t_end 1e-4 --number_of_samples 5000

python main.py --config cifar10_continuous.yml --sample_type unisampler --gpu 4 --exp exps/cifar10/nfe3_asf --sample --fid --load_decision /share/zoudongyun-nfs/DPM/afs_cifar10_4nfe.pth --statistics_dir dpm_solver_v3/cifar10_ddpmpp_deep_continuous/0.0001_1200_4096
```

## Solver Schedule Information
- `solver_schedules/cifar10_4nfe.pth`: NFE=4, FID=13.10
- `solver_schedules/cifar10_5nfe.pth`: NFE=5, FID=7.65
- `solver_schedules/cifar10_6nfe.pth`: NFE=6, FID=5.10
- `solver_schedules/cifar10_7nfe.pth`: NFE=7, FID=3.91
- `solver_schedules/cifar10_8nfe.pth`: NFE=8, FID=3.65
- `solver_schedules/cifar10_9nfe.pth`: NFE=9, FID=3.09
- `solver_schedules/cifar10_10nfe_improved.pth`: NFE=10, FID=2.69
