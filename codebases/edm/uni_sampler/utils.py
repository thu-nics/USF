import torch
import os
import yaml
import copy

def get_orders_and_timesteps_for_singlestep_solver(sampler, steps, order, skip_type, t_T, t_0, device):
    if order == 3:
        K = steps // 3 + 1
        if steps % 3 == 0:
            orders = [3,] * (K - 2) + [2, 1]
        elif steps % 3 == 1:
            orders = [3,] * (K - 1) + [1]
        else:
            orders = [3,] * (K - 1) + [2]
    elif order == 2:
        if steps % 2 == 0:
            K = steps // 2
            orders = [2,] * K
        else:
            K = steps // 2 + 1
            orders = [2,] * (K - 1) + [1]
    elif order == 1:
        K = steps
        orders = [1,] * steps
    else:
        raise ValueError("'order' must be '1' or '2' or '3'.")
    if skip_type == 'logSNR':
        # To reproduce the results in DPM-Solver paper
        timesteps_outer = sampler.get_time_steps(skip_type, t_T, t_0, K, device)
    else:
        timesteps_outer = sampler.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
    return timesteps_outer, orders

def get_empirical_decisions(args, sampler, device):
    '''
    Func: get all searchable decisions to match common sampler configurations. Decisions include: 1. taylor expansion orders; 2. timesteps; 3. prediction types; 4. derivative types; 5. start points; 6. corrector types; 7. use AFS or not;
    '''
    decision_type = args.uni_sampler_decision_type
    skip_type = args.skip_type
    lower_order_final = args.lower_order_final
    steps = args.timesteps
    t_0 = 1. / sampler.noise_schedule.total_N if getattr(args, "t_end", None) is None else getattr(args, "t_end", None)
    t_T = sampler.noise_schedule.T if getattr(args, "t_start", None) is None else getattr(args, "t_start", None)
    
    if decision_type in ["dpmsolver", "dpmsolver++"]: # get decisions which are equivalent with DPM-Solver
        order = args.dpm_solver_order
        method = args.dpm_solver_method # single step or multi step
        solver_type = args.dpm_solver_type # taylor or dpmsolver

        # get timesteps
        if method == "singlestep":
            outer_timesteps, outer_orders = get_orders_and_timesteps_for_singlestep_solver(sampler, steps, order, skip_type, t_T, t_0, device)
            timesteps = [outer_timesteps[0].item()]
            for index in range(len(outer_timesteps) - 1):
                outer_order = outer_orders[index]
                s, t = outer_timesteps[index], outer_timesteps[index + 1]
                inner_timesteps = sampler.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=outer_order, device=device)
                for inner_t in inner_timesteps[1:]:
                    timesteps.append(inner_t.item())
            timesteps = torch.tensor(timesteps).to(device)
        elif method == "multistep":
            timesteps = sampler.get_time_steps(skip_type, t_T, t_0, steps, device)
        
        # get orders
        orders = sampler.get_orders(order, method, lower_order_final, steps)
        
        # get prediction types
        if decision_type == "dpmsolver++":
            prediction_types = ["data_prediction"] * steps
        elif decision_type == "dpmsolver":
            prediction_types = ["noise_prediction"] * steps
        else:
            raise ValueError(f"algorithm_type must be dpmsolver or dpmsolver++! Got {decision_type}")
        
        # get start points
        start_points = sampler.get_start_points(method, orders)
            
        # get derivative types
        derivative_types = []
        if method == "singlestep":
            for index in range(len(orders)):
                i = orders[index]
                if i == 1:
                    derivative_types.append([{"estimate":f"Difference_{i - 1}", "relaxation":None, "active_points":[]}])
                elif i == 2:
                    if ((index < len(orders) - 1 and orders[index + 1] != 3) or index == len(orders) - 1) and solver_type == "dpmsolver":
                        derivative_relaxation = "dpmsolver-2"
                    else:
                        derivative_relaxation = None
                    derivative_type = []
                    for j in range(i - 1):
                        derivative_type.append({"estimate":f"Difference_{i - 1}", "relaxation":derivative_relaxation, "active_points":[-1]})
                    derivative_types.append(derivative_type)
                elif i == 3:
                    if solver_type == "dpmsolver":
                        derivative_types.append([{"estimate":f"Difference_{1}", "relaxation":None, "active_points":[-1]}] * 1)
                        orders[index] = 2
                    else:
                        derivative_type = []
                        for j in range(i - 1):
                            derivative_type.append({"estimate":f"Difference_{i - 1}", "relaxation":None, "active_points":[-1, -2]})
                        derivative_types.append(derivative_type)
        elif method == "multistep":
            for i in orders:
                if i == 2 and solver_type == "dpmsolver":
                    derivative_relaxation = "dpmsolver-2"
                else:
                    derivative_relaxation = None
                active_points = []
                for j in range(1, i):
                    active_points.append(- j - 1)
                derivative_type = []
                for j in range(i - 1):
                    derivative_type.append({"estimate":f"Difference_{i - 1}", "relaxation":derivative_relaxation, "active_points":copy.deepcopy(active_points)})
                derivative_types.append(derivative_type)
        global_derivative_relaxation_type = None
            
        # get corrector types
        corrector_types = [{"type": "no_corrector", "start_point": None, "taylor_order": None, "derivative_type": None}] * steps
        
    elif decision_type == "unipc":
        order = args.uni_pc_order
        prediction_type = args.uni_pc_prediction_type # noise_prediction or data_prediction
        variant = args.uni_pc_variant
        disable_corrector = args.uni_pc_disable_corrector
        method = "multistep" # currently only multistep is supported for UniPC
        
        # get timesteps
        timesteps = sampler.get_time_steps(skip_type, t_T, t_0, steps, device)
        
        # get orders
        orders = sampler.get_orders(order, method, lower_order_final, steps)
        
        # get prediction types
        prediction_types = [prediction_type] * steps
        
        # get start points
        start_points = sampler.get_start_points(method, orders)
        
        # get derivative types    
        derivative_types = []
        if variant == "vary_coeff":
            global_derivative_relaxation_type = "vary_coeff"
        elif variant == "bh1":
            global_derivative_relaxation_type = "unipc-bh1"
        elif variant == "bh2":
            global_derivative_relaxation_type = "dpmsolver-2"
        for order in orders:
            if order == 1:
                derivative_relaxation = None
                active_points = None
            if order == 2:
                if variant == "vary_coeff":
                    derivative_relaxation = None
                elif variant == "bh1":
                    derivative_relaxation = "unipc-bh1"
                elif variant == "bh2":
                    derivative_relaxation = "dpmsolver-2"
                active_points = [-2]
            elif order == 3:
                derivative_relaxation = None
                active_points = [-2, -3]
            elif order == 4:
                derivative_relaxation = None
                active_points = [-2, -3, -4]
            derivative_type = []
            for i in range(order - 1):
                derivative_type.append({"estimate":f"Difference_{order - 1}", "relaxation":derivative_relaxation, "active_points":active_points})
            derivative_types.append(derivative_type)
        
        # get corrector types
        if disable_corrector:
            corrector_types = []
            for j in range(steps):
                corrector_types.append({"type": "no_corrector", "start_point": None, "taylor_order": None, "derivative_type": None})
        else:
            corrector_types = []
            for i in range(steps - 1):
                correct_start_point, correct_taylor_order, correct_derivative_type = get_unipc_corrector_decisions(start_points[i], orders[i], derivative_types[i], None) # TODO: change the last args to global_derivative_relaxation_type after v5 search space generation
                corrector_types.append({"type": "pseudo", "start_point": correct_start_point, "taylor_order": correct_taylor_order, "derivative_type": correct_derivative_type})
            corrector_types.append({"type": "no_corrector", "start_point": None, "taylor_order": None, "derivative_type": None})
    
    elif decision_type == "dpmsolver_v3":
        order = args.dpmsolver_v3_order
        orders = sampler.get_orders(order, "multistep", lower_order_final, steps)
        start_points = sampler.get_start_points("multistep", orders)
        use_corrector = args.use_corrector
        p_pseudo = args.p_pseudo
        c_pseudo = args.c_pseudo
        t_end = 1. / sampler.noise_schedule.total_N if getattr(args, "t_end", None) is None else getattr(args, "t_end", None)
        t_start = sampler.noise_schedule.T if getattr(args, "t_start", None) is None else getattr(args, "t_start", None)
        timesteps = sampler.get_time_steps(skip_type, t_start, t_end, steps, device)
        indexes = sampler.convert_to_ems_indexes(timesteps)
        timesteps = sampler.convert_to_timesteps(indexes,device=device)
        prediction_types = ["dpmsolver_v3_prediction"] * steps
        # get derivative types
        derivative_types = get_dpmsolver_v3_derivative_types(orders, p_pseudo)
        corrector_types = []
        if use_corrector:
            c_pseudo = args.c_pseudo
            for i in range(steps - 1):
                correct_start_point, correct_taylor_order, correct_derivative_type = get_dpmsolver_v3_corrector_decisions(start_points[i], orders[i], c_pseudo)
                if correct_taylor_order > 1:
                    corrector_types.append({"type": "pseudo", "start_point": correct_start_point, "taylor_order": correct_taylor_order, "derivative_type": correct_derivative_type})
                else:
                    corrector_types.append({"type": "no_corrector", "start_point": None, "taylor_order": None, "derivative_type": None})
            corrector_types.append({"type": "no_corrector", "start_point": None, "taylor_order": None, "derivative_type": None})
        else:
            corrector_types = [{"type": "no_corrector", "start_point": None, "taylor_order": None, "derivative_type": None}] * steps
        
    else:
        raise NotImplementedError
    
    afs = args.afs if hasattr(args, "afs") else "no_afs"
    skip_coefficients = torch.linspace(1.0, 1.0, 15)
    return {
        "timesteps": timesteps,
        "orders": orders,
        "prediction_types": prediction_types,
        "start_points": start_points,
        "derivative_types": derivative_types,
        "corrector_types": corrector_types, 
        "afs": afs,
        "skip_coefficients": skip_coefficients
    }

def get_dpmsolver_v3_derivative_types(orders, pseudo):
    derivative_types = []
    order_active_points_dict = {1: None, 2: [-2], 3: [-2, -3], 4: [-2, -3, -4]}
    if pseudo:
        for order in orders:
            derivative_type = []
            for i in range(2, order + 1):
                derivative_type.append({"estimate":f"Difference_{i-1}", "relaxation":None, "active_points":order_active_points_dict[i]})
            derivative_types.append(derivative_type)
    else:
        for order in orders:
            active_points = order_active_points_dict[order]
            derivative_type = []
            for i in range(1, order):
                derivative_type.append({"estimate":f"Difference_{order - 1}", "relaxation":None, "active_points":active_points})
            derivative_types.append(derivative_type)
    return derivative_types

def get_unipc_corrector_decisions(start_point, taylor_order, derivative_type, global_derivative_relaxation_type):
    correct_start_point = start_point - 1
    correct_taylor_order = taylor_order + 1
    correct_derivative_types = []
    if len(derivative_type) > 0:
        # adjust the estimating method of existing derivatives
        import copy
        for index, curr_derivative in enumerate(derivative_type):
            correct_derivative = copy.deepcopy(curr_derivative)
            if correct_derivative["active_points"] is not None:
                for ap_index in range(len(correct_derivative["active_points"])):
                    correct_derivative["active_points"][ap_index] -= 1
                correct_derivative["active_points"].append(-1)
            estimate_order = int(curr_derivative["estimate"].split("_")[1]) + 1
            correct_derivative["estimate"] = f"Difference_{estimate_order}"
            correct_derivative["relaxation"] = None
            correct_derivative_types.append(correct_derivative)
        # add an additional derivative
        correct_derivative_types.append({"estimate": f"Difference_{estimate_order}", "active_points": correct_derivative["active_points"], "relaxation":None})
    else: # the current update is DDIM update
        assert taylor_order == 1
        correct_derivative_types.append({"estimate": "Difference_1", "active_points": [-1], "relaxation":global_derivative_relaxation_type}) 
    return correct_start_point, correct_taylor_order, correct_derivative_types

def get_dpmsolver_v3_corrector_decisions(start_point, taylor_order, c_pseudo):
    if c_pseudo:
        correct_start_point = start_point - 1
        correct_taylor_order = taylor_order + 1
        correct_derivative_types = []
        order_active_points_dict = {1: None, 2: [-1], 3: [-1, -3], 4: [-1, -3, -4]}
        # adjust the estimating method of existing derivatives
        for i in range(2, correct_taylor_order+1):
            correct_derivative_types.append({"estimate":f"Difference_{i-1}", "relaxation":None, "active_points":order_active_points_dict[i]})
        return correct_start_point, correct_taylor_order, correct_derivative_types
    else:
        # full order estimation
        correct_start_point = start_point - 1
        correct_taylor_order = taylor_order
        correct_derivative_types = []
        if correct_taylor_order == 1:
            # the current update is DDIM update
            correct_derivative_types.append({"type": "no_corrector", "start_point": None, "taylor_order": None, "derivative_type": None})
        else:
            order_active_points_dict = {1: None, 2: [-1], 3: [-1, -3], 4: [-1, -3, -4]}
        # adjust the estimating method of existing derivatives
        for i in range(2, correct_taylor_order+1):
            correct_derivative_types.append({"estimate":f"Difference_{correct_taylor_order-1}", "relaxation":None, "active_points":order_active_points_dict[correct_taylor_order]})
        return correct_start_point, correct_taylor_order, correct_derivative_types

def print_decisions(logger, decisions):
    def info(logger, s):
        if logger is None:
            print(s)
        else:
            logger.info(s)
    for key in decisions.keys():
        info(logger, f"{key}:")
        info(logger, decisions[key])
        info(logger, "-"*50)
    
