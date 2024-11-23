import torch
import numpy as np
import random
import pathlib
import copy

def get_population(base_path, search_config):
    base_path = pathlib.Path(base_path)
    data_paths = [file for file in base_path.glob('**/*.{}'.format("pth"))]
    nfe_num = random.randint(search_config["total_timesteps_range"]["min"], search_config["total_timesteps_range"]["max"])
    print(f"Total nfe: {nfe_num}")
    population = []
    for path in data_paths:
        mutate = torch.load(path)[0]
        if get_nfe(mutate) == nfe_num:
            population.append(torch.load(path))
    print(f"Population size: {len(population)}")
    population = sorted(population, key=lambda x: x[1] * search_config["metric"].get("indicator", 1))
    
    return population

def select_parents(population, search_config, num=1):
    if random.uniform(0, 1) < search_config["parents"]["rank_prob"]:
        mode = "rank"
    else:
        mode = "absolute"

    if mode == "rank":
        all_candidate_parents = population[:search_config["parents"]["rank_bar"]]
    elif mode == "absolute":
        all_candidate_parents = []
        for p in population[1:]:
            if abs(p[1] - population[0][1]) < search_config["parents"]["absolute_bar"]:
                all_candidate_parents.append(p)
            else:
                break
    
    selected_parents = random.sample(all_candidate_parents, num)
    if num == 1:
        return selected_parents[0]
    else:
        return selected_parents

# def get_baseline(runner):
#     pass

def crossover(decision_1:dict, decision_2:dict,config)->dict:
    if len(decision_1["orders"]) != len(decision_2["orders"]):
        print("The length of the two parents decision is not the same! Can't crossover! Randomly return one of them")
        return random.sample([decision_1, decision_2], 1)[0]
    expression_from = {}
    full_key = list(decision_1.keys())
    for key in full_key:
        expression_from[key] = random.randint(0, 1)
    if not 0 in expression_from.values():
        expression_from[random.sample(full_key, 1)[0]] = 0
    if not 1 in expression_from.values():
        expression_from[random.sample(full_key, 1)[0]] = 1
    expression_from["derivative_types"] = expression_from["orders"]
    new_decision = {}
    for key in expression_from.keys():
        if expression_from[key] == 0:
            parents_from = decision_1
        else:
            parents_from = decision_2
        new_decision[key] = parents_from[key]
    for i in range(len(new_decision["orders"])):
        if new_decision["corrector_types"][i]["type"] != "no_corrector":
            corrector_type = get_corrector_type(new_decision["corrector_types"][i]["type"], -1, new_decision["orders"][i], new_decision["derivative_types"][i])
            new_decision["corrector_types"][i] = corrector_type
    if random.uniform(0, 1) < config["crossover"]["deep_crossover_prob"]:
        # crossover the prediction types and orders
        new_prediction_types = []
        new_orders = []
        new_derivative_types = []
        for i in range(len(decision_1["prediction_types"])):
            if random.uniform(0, 1) < 0.5:
                new_prediction_types.append(decision_1["prediction_types"][i])
                new_orders.append(decision_1["orders"][i])
                new_derivative_types.append(decision_1["derivative_types"][i])
            else:
                new_prediction_types.append(decision_2["prediction_types"][i])
                new_orders.append(decision_2["orders"][i])
                new_derivative_types.append(decision_2["derivative_types"][i])
        new_decision["prediction_types"] = new_prediction_types
        new_decision["orders"] = new_orders
        new_decision["derivative_types"] = new_derivative_types
    return new_decision

def mutate(decision, config, budget=None):
    
    id_correct(decision)
    
    new_decision = copy.deepcopy(decision)
    original_budget = len(decision["orders"])
    for corrector_type in decision["corrector_types"]:
        if corrector_type["type"] == "implicit":
            original_budget += 1
    if decision["corrector_types"][-1] == "pseudo":
        original_budget += 1
    
    p = random.uniform(0, 1)
    if budget is None or original_budget == budget:
        step_increase_prob = config["mutate"]["step_increase"]
        step_decrease_prob = config["mutate"]["step_decrease"]
    else:
        assert original_budget < budget
        step_increase_prob = 1.0
        step_decrease_prob = 0.0
    if p < step_increase_prob:
        # get timestep for the new step
        new_t = random.uniform(config["mutate"]["timesteps"]["t_end_list"][0], config["mutate"]["timesteps"]["t_start"])
        new_timesteps = new_decision["timesteps"].cpu().detach().numpy().tolist()
        new_timesteps += [new_t]
        new_timesteps.sort(reverse=True)
        new_step_index = new_timesteps.index(new_t) - 1
        new_decision["timesteps"] = torch.tensor(new_timesteps)
        # get order for the new step
        new_order = get_decision_from_dict(config["mutate"]["taylor_orders"]["order_list"])
        new_order = min(new_order, new_step_index + 1)
        new_decision["orders"].insert(new_step_index, new_order)
        # get prediction type for the new step
        new_prediction_type = get_decision_from_dict(config["mutate"]["prediction_types"]["type_list"])
        new_decision["prediction_types"].insert(new_step_index, new_prediction_type)
        # get start point for the new step
        new_decision["start_points"].insert(new_step_index, -1)
        # get derivative type for the new step
        derivative_type = []
        for j in range(new_order - 1):
            if j == 0:
                relaxation_coefficient = 0
                relaxation_type = "linear"
                p = random.uniform(0, 1)
                if p < config["mutate"]["derivative_estimation_order"]["evo_prob"]:
                    active_points_num = min(get_decision_from_dict(config["mutate"]["derivative_estimation_order"]["order_list"]), new_order - 1)
                    active_points = list(range(- active_points_num - 1, -1))
                else:
                    active_points = list(range(- new_order, -1))
            else:
                relaxation_type = None
                active_points = list(range(- new_order, -1))
            if j == 0:
                derivative_type.append({"estimate": f"Difference_{len(active_points)}", "active_points": active_points, "relaxation":relaxation_type,"relaxation_coefficient":relaxation_coefficient})
            else:
                derivative_type.append({"estimate": f"Difference_{len(active_points)}", "active_points": active_points, "relaxation":relaxation_type})
        new_decision["derivative_types"].insert(new_step_index, derivative_type)
        # get corrector type for the new step
        corrector_type = get_decision_from_dict(config["mutate"]["corrector"]["type_list"])
        corrector_type = get_corrector_type(corrector_type, -1, new_order, derivative_type)
        new_decision["corrector_types"].insert(new_step_index, corrector_type)
    elif p > 1 - step_decrease_prob:
        delete_index = random.randint(1, len(new_decision["orders"]) - 1)
        # delete the timestep
        new_timesteps = new_decision["timesteps"].cpu().detach().numpy().tolist()
        new_timesteps.pop(delete_index)
        new_decision["timesteps"] = torch.tensor(new_timesteps)
        # delete the order
        new_decision["orders"].pop(delete_index - 1)
        # delete the prediction type
        new_decision["prediction_types"].pop(delete_index - 1)
        # delete the start point
        new_decision["start_points"].pop(delete_index - 1)
        # delete the derivative type
        new_decision["derivative_types"].pop(delete_index - 1) 
        # delete the corrector type
        new_decision["corrector_types"].pop(delete_index - 1)
        # fix the first several orders
        for i in range(len(new_decision["orders"])):
            if new_decision["orders"][i] > i + 1:
                new_decision["orders"][i] = i + 1
                derivative_type = []
                for j in range(i):
                    if j == 0:
                        relaxation_coefficient = new_decision["derivative_types"][i][j].get("relaxation_coefficient", 0)
                        relaxation_type = "linear"
                        p = random.uniform(0, 1)
                        if p < config["mutate"]["derivative_estimation_order"]["evo_prob"]:
                            active_points_num = min(get_decision_from_dict(config["mutate"]["derivative_estimation_order"]["order_list"]), i)
                            active_points = list(range(- active_points_num - 1, -1))
                        else:
                            active_points = list(range(- i - 1, -1))
                    else:
                        relaxation_type = None
                        active_points = list(range(- i - 1, -1))
                    if j == 0:
                        derivative_type.append({"estimate": f"Difference_{len(active_points)}", "active_points": active_points, "relaxation":relaxation_type,"relaxation_coefficient":relaxation_coefficient})
                    else:
                        derivative_type.append({"estimate": f"Difference_{len(active_points)}", "active_points": active_points, "relaxation":relaxation_type})
                new_decision["derivative_types"][i] = derivative_type
                c_t = get_decision_from_dict(config["mutate"]["corrector"]["type_list"])
                corrector_type = get_corrector_type(c_t, -1, i + 1, derivative_type)
                new_decision["corrector_types"][i] = corrector_type
    else:
        pass
    
    for i in range(len(new_decision["orders"])):
        # mutate timestep
        p = random.uniform(0, 1)
        if p < config["mutate"]["timesteps"]["evo_prob"]:
            if i != len(new_decision["orders"]) - 1:
                mu = 0
                sigma = config["mutate"]["timesteps"]["evo_sigma"]
                delta = random.gauss(mu, sigma)
                new_decision["timesteps"][i + 1] += delta
                if new_decision["timesteps"][i + 1] > new_decision["timesteps"][0] - 1e-3 or new_decision["timesteps"][i + 1] < new_decision["timesteps"][-1] + 1e-3:
                    new_decision["timesteps"][i + 1] -= delta
                if new_decision["timesteps"][i + 1] < config["mutate"]["timesteps"]["t_end_list"][0]:
                    new_decision["timesteps"][i + 1] = config["mutate"]["timesteps"]["t_end_list"][0]
                if new_decision["timesteps"][i + 1] > config["mutate"]["timesteps"]["t_start"]:
                    new_decision["timesteps"][i + 1] = config["mutate"]["timesteps"]["t_start"]
            else:
                new_decision["timesteps"][i + 1] = random.uniform(config["mutate"]["timesteps"]["t_end_list"][0], config["mutate"]["timesteps"]["t_end_list"][1])
        # mutate order
        p = random.uniform(0, 1)
        if p < config["mutate"]["taylor_orders"]["evo_prob"]:
            new_order = min(get_decision_from_dict(config["mutate"]["taylor_orders"]["order_list"]), i + 1)
            if new_decision["orders"][i] != new_order:
                new_decision["orders"][i] = new_order
                derivative_type = []
                for j in range(new_order - 1):
                    derivative_type.append({"estimate": f"Difference_{new_order - 1}", "active_points": list(range(- new_order, -1)), "relaxation":None})
                new_decision["derivative_types"][i] = derivative_type
        # mutate prediction type
        p = random.uniform(0, 1)
        if p < config["mutate"]["prediction_types"]["evo_prob"]:
            new_decision["prediction_types"][i] = get_decision_from_dict(config["mutate"]["prediction_types"]["type_list"])
        # mutate derivative relaxtion type and estimation order
        if new_decision["orders"][i] > 1:
            for j in range(len(new_decision["derivative_types"][i])):
                # relaxation type
                p = random.uniform(0, 1)
                if p < config["mutate"]["derivative_relaxation"]["evo_prob"] and j == 0:
                    new_decision["derivative_types"][i][j]["relaxation"] = "linear"
                    relaxation_coefficient = new_decision["derivative_types"][i][j].get("relaxation_coefficient", 0)
                    if random.uniform(0, 1) < config["mutate"]["derivative_relaxation"]["evo_prob"]:
                        mu = 0
                        sigma = config["mutate"]["derivative_relaxation"]["evo_sigma"]
                        delta = random.gauss(mu, sigma)
                        relaxation_coefficient += delta
                    new_decision["derivative_types"][i][j]["relaxation_coefficient"] = relaxation_coefficient
                # estimation order
                p = random.uniform(0, 1)
                if p < config["mutate"]["derivative_estimation_order"]["evo_prob"] and j == 0:
                    active_points_num = min(get_decision_from_dict(config["mutate"]["derivative_estimation_order"]["order_list"]), new_decision["orders"][i] - 1)
                    new_decision["derivative_types"][i][j]["active_points"] = list(range(- active_points_num - 1, -1))
                    new_decision["derivative_types"][i][j]["estimate"] = f"Difference_{active_points_num}"
        # mutate corrector type
        p = random.uniform(0, 1)
        if p < config["mutate"]["corrector"]["evo_prob"]:
            c_t = get_decision_from_dict(config["mutate"]["corrector"]["type_list"])
        else:
            c_t = new_decision["corrector_types"][i]["type"]
        corrector_type = get_corrector_type(c_t, -1, new_decision["orders"][i], new_decision["derivative_types"][i])
        new_decision["corrector_types"][i] = corrector_type
            
    new_timesteps = new_decision["timesteps"].cpu().detach().numpy().tolist()
    new_timesteps.sort(reverse=True)
    new_decision["timesteps"] = torch.tensor(new_timesteps)
    
    c_t = get_decision_from_dict(config["mutate"]["corrector"]["final_type_list"])
    corrector_type = get_corrector_type(c_t, new_decision["start_points"][-1], new_decision["orders"][-1], new_decision["derivative_types"][-1])
    new_decision["corrector_types"][-1] = corrector_type
    # analytical first step
    afs_evo_prob = config["mutate"]["afs"]["evo_prob"]
    p_afs = random.uniform(0, 1)
    if p_afs < afs_evo_prob:
        new_decision["afs"] = get_decision_from_dict(config["mutate"]["afs"]["afs_list"])
    # U-net skip coefficient
    skip_evo_prob = config["mutate"]["skip"]["evo_prob"]
    p_skip = random.uniform(0, 1)
    if p_skip < skip_evo_prob:
        # skip coefficients is a 1d tensor
        skip_coefficients = new_decision["skip_coefficients"]
        mu = config["mutate"]["skip"]["evo_mu"]
        sigma = config["mutate"]["skip"]["evo_sigma"]
        delta = torch.normal(mu, sigma, skip_coefficients.size())
        skip_coefficients += delta
        skip_coefficients = torch.clamp(skip_coefficients, 0, 1)
        new_decision["skip_coefficients"] = skip_coefficients
    return new_decision

def equal(decision1, decision2):
    if len(decision1["timesteps"]) != len(decision2["timesteps"]):
        return False
    if not (decision1["timesteps"] == decision2["timesteps"]).all():
        return False
    for key in decision1.keys():
        if key != "timesteps":
            if decision1[key] != decision2[key]:
                return False

    return True

def get_nfe(decision):
    nfe = len(decision["orders"])
    if decision.get("afs", "no_afs") != "no_afs":
        nfe -= 1
    for corrector_type in decision["corrector_types"]:
        if corrector_type["type"] == "implicit":
            nfe += 1
    return nfe

def get_decision_from_dict(prob_dict):

    if len(prob_dict.values()) > 0:
        assert 1 - 1e-6 <= sum(prob_dict.values()) <= 1 + 1e-6, f"The sum of probabilities in dict {prob_dict} must be 1!"
    p = random.uniform(0, 1)
    temp = 0
    for key in prob_dict.keys():
        temp += prob_dict[key]
        if temp > p:
            return key
    return None

def get_decision_list_from_dict(nfe, prob_dict):
    
    mutate = []
    for _ in range(nfe):
        mutate.append(get_decision_from_dict(prob_dict))

    return mutate

def get_random_timesteps(nfe, transform_list, t_end_list, t_start=1., eps=1e-3):
    
    transform_name = get_decision_from_dict(transform_list)
    
    if transform_name == "uniform":
        transform = lambda x: x
    elif transform_name == "quadratic_back":
        transform = lambda x: x ** 0.5
    elif transform_name == "quadratic_front":
        transform = lambda x: 1 - x ** 0.5
    else:
        raise NotImplementedError
    
    t_end = random.uniform(t_end_list[0], t_end_list[1])
    
    timesteps = [t_end, t_start]
    for _ in range(nfe - 1):
        timestep = transform(random.uniform(t_end, t_start))
        timestep = max(t_end + eps, timestep)
        timestep = min(t_start - eps, timestep)
        timesteps.append(timestep)
    timesteps.sort(reverse=True)
    timesteps = torch.tensor(timesteps)
    for i in range(nfe - 1, 0, -1):
        timesteps[i] = max(timesteps[i + 1] + eps, timesteps[i])
    
    return timesteps

def get_corrector_mutate(p_start_point, p_taylor_order, p_derivative_type):
    correct_start_point = p_start_point - 1
    correct_taylor_order = p_taylor_order + 1
    correct_derivative_types = []
    if len(p_derivative_type) > 0 and p_taylor_order != 1:
        # adjust the estimating method of existing derivatives
        import copy
        for index, curr_derivative in enumerate(p_derivative_type):
            correct_derivative = copy.deepcopy(curr_derivative)
            if correct_derivative["active_points"] is not None:
                for ap_index in range(len(correct_derivative["active_points"])):
                    correct_derivative["active_points"][ap_index] -= 1
                correct_derivative["active_points"].append(-1)
            estimate_order = int(curr_derivative["estimate"].split("_")[1]) + 1
            correct_derivative["estimate"] = f"Difference_{estimate_order}"
            correct_derivative["relaxation"] = None # or curr_derivative[0]["relaxation"] ?
            correct_derivative_types.append(correct_derivative)
        # add an additional derivative
        correct_derivative_types.append({"estimate": f"Difference_{estimate_order}", "active_points": correct_derivative["active_points"], "relaxation": None})
    else: # the current update is DDIM update
        assert p_taylor_order == 1
        correct_derivative_types.append({"estimate": "Difference_1", "active_points": [-1], "relaxation": None}) 
    return correct_start_point, correct_taylor_order, correct_derivative_types

def get_corrector_type(t, start_point, taylor_order, derivative_type):
    if t == "no_corrector":
        corrector_type = {'type': 'no_corrector', 'start_point': None, 'taylor_order': None, 'derivative_type': None}
    else:
        assert t in ["pseudo", "implicit"]
        c_start_point, c_taylor_order, c_derivative_type = get_corrector_mutate(start_point, taylor_order, derivative_type)
        corrector_type = {"type": t, "start_point": c_start_point, "taylor_order": c_taylor_order, "derivative_type": c_derivative_type}
    
    return corrector_type

def id_correct(decision):
    if "derivative_types" in decision.keys():
        for i in range(len(decision["derivative_types"])):
            for j in range(len(decision["derivative_types"][i])):
                decision["derivative_types"][i][j] = copy.deepcopy(decision["derivative_types"][i][j])