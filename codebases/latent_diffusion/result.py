import torch
import pathlib

def main():
    # result_path = "./search_data/initialize"
    result_path = "./search_data"

    base_path = pathlib.Path(result_path)
    data_paths = [file for file in base_path.glob('**/*.{}'.format("pth"))]

    result = {}
    best_decisions = {}
    nfe_range = range(3,11)
    for nfe in nfe_range:
        result[nfe] = []
        best_decisions[nfe] = None

    for path in data_paths:
        decision_and_fid = torch.load(path,map_location="cpu")
        # if decision_and_fid[0]["afs"] != "no_afs":
        #     continue
        nfe = get_nfe(decision_and_fid[0])
        if nfe in nfe_range:
            result[nfe].append(decision_and_fid)
    
    for nfe in nfe_range:
        # 获取最小fid对应的decision
        min_fid = float("inf")
        min_fid_decision = None
        for decision_and_fid in result[nfe]:
            if decision_and_fid[1] < min_fid:
                min_fid = decision_and_fid[1]
                min_fid_decision = decision_and_fid[0]
        best_decisions[nfe] = min_fid_decision
        prediction_types = min_fid_decision["prediction_types"]
        afs = min_fid_decision.get("afs", "no_afs")
        print(f"nfe={nfe}, min_fid={min_fid}, afs={afs}")
        # print(f"afs:{afs}")
        # print(f"prediction_types={prediction_types}, afs={afs}")
        torch.save(min_fid_decision,f"solver_schedules/10.17/stable_diffusion_nfe{nfe}.pth")

def get_nfe(decision):
    nfe = len(decision["orders"])
    if decision.get("afs", "no_afs") != "no_afs":
        nfe -= 1
    for corrector_type in decision["corrector_types"]:
        if corrector_type["type"] == "implicit":
            nfe += 1
    return nfe


if __name__ == "__main__":
    main()