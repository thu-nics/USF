import torch
from torch.utils import data
import os
import json
import copy

class PromptNoiseJointSet(data.Dataset):
    def __init__(self, text, noise):
        assert len(noise.shape)==4
        if len(text)!=len(noise):
            cut_length = min(len(text), len(noise))
            text = text[:cut_length]
            noise = noise[:cut_length]
        self.text = text
        self.noise = noise
    
    def __len__(self):
        return len(self.noise)
    
    def __getitem__(self, index):
        return self.text[index], self.noise[index]
    
def prepare_noise(shape, use_fixed_noise, fixed_noise_path, save_as_fixed_noise=False):
    if use_fixed_noise and os.path.exists(fixed_noise_path):
        noise = torch.load(fixed_noise_path)
    else:
        noise = torch.randn(shape)
    if save_as_fixed_noise:
        torch.save(noise, fixed_noise_path)
    return noise

def prepare_text(prompt_file):
    with open(prompt_file, "r") as f:
        prompts = f.read().splitlines()
    return prompts

def prepare_coco_text_and_image(json_file):
    info = json.load(open(json_file, 'r'))
    annotation_list = info["annotations"]
    image_caption_dict = {}
    for annotation_dict in annotation_list:
        if annotation_dict["image_id"] in image_caption_dict.keys():
            image_caption_dict[annotation_dict["image_id"]].append(annotation_dict["caption"])
        else:
            image_caption_dict[annotation_dict["image_id"]] = [annotation_dict["caption"]]
    captions = list(image_caption_dict.values())
    image_ids = list(image_caption_dict.keys())
    
    active_captions = []
    for texts in captions:
        active_captions.append(texts[0])
        
    image_paths = []
    for image_id in image_ids:
        image_paths.append("/mnt/sharedata/les19/dataset/coco/val2014/"+f"COCO_val2014_{image_id:012}.jpg")
    return active_captions, image_paths

def index_preprocess(
    set, 
    specified_ids: str,
    eval_num: int, 
    image_set=None,
    proxy_index_type="linear", 
    interval_begin=None,
    load_index=None,
    continuous_sample=False, 
    pre_cut=True, # Cut coco-validation set to 30000 images.
    pre_cut_length=30000,
    base_count=0,
    sample_path=None,
):
    #-------------------------------If image ids are specified---------------------------------#
    if specified_ids is not None:
        ids = torch.load(specified_ids)
        new_image_set = []
        new_text = []
        new_noise = []
        for id in ids:
            for index, path in enumerate(image_set):
                if f"{id}.jpg" in path:
                    new_image_set.append(path)
                    new_noise.append(set.noise[index])
                    new_text.append(set.text[index])
                    break
        set.text = new_text
        set.noise = torch.stack(new_noise, dim=0)
        return set, new_image_set
                    
    #-------------------------------------Pre-process-----------------------------------------#
    if pre_cut: 
        if image_set is not None:   
            new_image_set = image_set[:pre_cut_length]
        if hasattr(set, "text"):
            set.text = set.text[:pre_cut_length]
        if hasattr(set, "noise"):
            set.noise = set.noise[:pre_cut_length]
        full_length = pre_cut_length
    else:
        full_length = None
        if hasattr(set, "text"):    
            full_length = len(set.text)
        if hasattr(set, "noise"):
            if full_length is not None:
                assert len(set.noise)==full_length
            else:
                full_length = len(set.noise)
        if image_set is not None:
            if full_length is not None:
                assert len(image_set)==full_length
            else:
                full_length = len(image_set)
    #-----------------------------------------------------------------------------------------#
        
    #------------------------------------Proxy index choice-----------------------------------#
    if proxy_index_type=="linear":
        if image_set is not None:
            new_image_set = image_set[0:full_length:full_length//eval_num][:eval_num]   
            image_set = new_image_set 
        if hasattr(set, "text"):
            new_text = set.text[0:full_length:full_length//eval_num][:eval_num]    
            set.text = new_text
        if hasattr(set, "noise"):
            new_noise = set.noise[0:full_length:full_length//eval_num][:eval_num] 
            set.noise = new_noise
    elif proxy_index_type=="front":
        if image_set is not None:
            image_set = image_set[:eval_num]    
        if hasattr(set, "text"):
            set.text = set.text[:eval_num]    
        if hasattr(set, "noise"):
            set.noise = set.noise[:eval_num]    
    elif proxy_index_type=="interval":
        assert interval_begin is not None
        if image_set is not None:
            image_set = image_set[interval_begin:interval_begin+eval_num]    
        if hasattr(set, "text"):
            set.text = set.text[interval_begin:interval_begin+eval_num]    
        if hasattr(set, "noise"):
            set.noise = set.noise[interval_begin:interval_begin+eval_num]
    elif proxy_index_type=="random":
        from random import sample
        new_index = list(range(0, full_length), eval_num).sort()
        if image_set is not None:
            new_image_set = []
            for index in new_index:
                new_image_set.append(image_set[index])
            image_set = new_image_set
        if hasattr(set, "text"):
            new_text = []
            for index in new_index:
                new_text.append(set.text[index])
            set.text = new_text
        if hasattr(set, "noise"):
            new_noise = torch.zeros([eval_num, set.noise.shape[1], set.noise.shape[2], set.noise.shape[3]])
            for step, index in enumerate(new_index):
                new_noise[step] = set.noise[index]
            set.noise = new_noise
    elif proxy_index_type=="load":
        assert load_index is not None
        indices = torch.load(load_index)
        if image_set is not None:
            new_image_set = []
            for index in indices:
                new_image_set.append(image_set[index])
            image_set = new_image_set
        if hasattr(set, "text"):
            new_text = []
            for index in indices:
                new_text.append(set.text[index])
            set.text = new_text
        if hasattr(set, "noise"):
            new_noise = torch.zeros([eval_num, set.noise.shape[1], set.noise.shape[2], set.noise.shape[3]])
            for step, index in enumerate(indices):
                new_noise[step] = set.noise[index]
            set.noise = new_noise
    else:
        raise NotImplementedError
    #-----------------------------------------------------------------------------------------#
    
    #-----------------------------------continuous Sample-------------------------------------#
    set.ori_text = set.text
    if continuous_sample is not None:
        assert sample_path is not None
        assert base_count == len(os.listdir(sample_path))  
        if hasattr(set, "text"):
            set.text = copy.deepcopy(set.text)[base_count:]   
        if hasattr(set, "noise"):
            set.noise = set.noise[base_count:]  
    #-----------------------------------------------------------------------------------------#

    return set, image_set

def get_coco_statistic_path(
        eval_num,
        proxy_index_type,
        pre_cut,
        pre_cut_length,
        interval_begin="",
    ):
    path = "/home/zhaotianchen/project/diffusion/les19_data/temp_files/activations/coco/"
    if pre_cut:
        path += f"total{pre_cut_length}"
    path += f"proxy{proxy_index_type}"
    if proxy_index_type=="":
        path += f"_start{interval_begin}"
    path += f"_eval{eval_num}.pth" 
    return path