#!/usr/bin/env python
########################################################################
# import default python-library
########################################################################
import os
import sys
import glob

import numpy
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
# from import
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn

from utils import *
from model import TorchModel
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################

S1 = 'id_04'
S2 = 'id_06'
MACHINE = 'slider'
machine_types = [S1, S2]
num_eval_normal = 250
force_high_overlap = True

########################################################################
# feature extractor
########################################################################

def shift(data, amount_to_right):
    amount_to_right = int(amount_to_right)
    if amount_to_right > 0:
        new_data = np.concatenate([np.zeros_like(data[:amount_to_right]), data[:-amount_to_right]])
    elif amount_to_right < 0:
        new_data = np.concatenate([data[amount_to_right:], np.zeros_like(data[:amount_to_right])])
    else:
        new_data = data
    return new_data

def train_file_to_mixture_wav(filename):
    shift_amount = [0.1, 0.3, 0.5]
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0
    src_filename_lst = []
    mix_label = None
    
    for machine in machine_types:
        src_filename = filename.replace(machine_type, machine)
        src_filename_lst.append(src_filename)
        sr, y = demux_wav(src_filename)
        label, spec_label = generate_label(np.expand_dims(y, axis=0), MACHINE)
        
        if force_high_overlap:
            if mix_label == None:
                mix_label = label
                ys = ys + y
            else:
                y_candidates = [shift(y, -shift_amount[2]* sr), 
                                shift(y, -shift_amount[1] * sr), 
                                shift(y, -shift_amount[0] * sr), 
                                y, 
                                shift(y, shift_amount[0] * sr),
                                shift(y, shift_amount[1] * sr),
                                shift(y, shift_amount[2] * sr),
                                ]
                label_candidates = list(map(lambda x: generate_label(np.expand_dims(x, axis=0), MACHINE)[0], y_candidates))
                overlap_candidates = list(map(lambda x: np.logical_and(x, mix_label).sum() / np.logical_or(x, mix_label).sum(), label_candidates))
                
                target_idx = np.argmax(overlap_candidates)
                label = label_candidates[target_idx]
                mix_label = np.logical_or(mix_label, label)
                ys = ys + y_candidates[target_idx]
        else:
            ys = ys + y

    return sr, ys, src_filename_lst
    
def eval_file_to_mixture_wav(filename):
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0
    for normal_type in machine_types:
        if normal_type == machine_type:
            src_filename = filename
        else:
            src_filename = filename.replace(machine_type, normal_type).replace('abnormal', 'normal')
        sr, y = demux_wav(src_filename)
        ys = ys + y
    
    return sr, ys

def eval_file_to_mixture_wav_label(filename):
    shift_amount = [0.1, 0.3, 0.5]
    machine_type = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    ys = 0
    mix_label = None
    gt_wav = {}
    active_label_sources = {}
    active_spec_label_sources = {}
    for normal_type in machine_types:
        if normal_type == machine_type:
            src_filename = filename
        else:
            src_filename = filename.replace(machine_type, normal_type).replace('abnormal', 'normal')
        sr, y = demux_wav(src_filename)
        if force_high_overlap:
            if mix_label == None:
                label, spec_label = generate_label(np.expand_dims(y, axis=0), MACHINE)
                mix_label = label
            else:
                y_candidates = [shift(y, -shift_amount[2]* sr), 
                                shift(y, -shift_amount[1] * sr), 
                                shift(y, -shift_amount[0] * sr), 
                                y, 
                                shift(y, shift_amount[0] * sr),
                                shift(y, shift_amount[1] * sr),
                                shift(y, shift_amount[2] * sr),
                                ]
                label_candidates = list(map(lambda x: generate_label(np.expand_dims(x, axis=0), MACHINE)[0], 
                                       y_candidates))
                overlap_candidates = list(map(lambda x: np.logical_and(x, mix_label).sum() / np.logical_or(x, mix_label).sum(), 
                                       label_candidates))
                target_idx = np.argmax(overlap_candidates)
                label = label_candidates[target_idx]
                mix_label = np.logical_or(mix_label, label)
                y = y_candidates[target_idx]
                
        ys = ys + y
        # label, spec_label = generate_label(np.expand_dims(y, axis=0), MACHINE)
        active_label_sources[normal_type] = label
        active_spec_label_sources[normal_type] = spec_label
        gt_wav[normal_type] = y
    
    return sr, ys, gt_wav, active_label_sources, active_spec_label_sources

def train_list_to_mixture_spec_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):

        sr, ys, src_file_lst = train_file_to_mixture_wav(file_list[idx])
        _, y1 = demux_wav(src_file_lst[0])
        _, y2 = demux_wav(src_file_lst[1])
        _, label_spec1 = generate_label(numpy.expand_dims(y1, axis=0), MACHINE)
        _, label_spec2 = generate_label(numpy.expand_dims(y2, axis=0), MACHINE)
        spec_label1 = label_spec1[:1, :, :].unsqueeze(3).repeat(1, 1, 1, n_mels).reshape(1, 309, frames * n_mels).squeeze(0).numpy()
        spec_label2 = label_spec2[:1, :, :].unsqueeze(3).repeat(1, 1, 1, n_mels).reshape(1, 309, frames * n_mels).squeeze(0).numpy()
        
        vector_array1 = wav_to_spec_vector_array(sr, ys,
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power,
                                            spec_mask=spec_label1)
        
        vector_array2 = wav_to_spec_vector_array(sr, ys,
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power,
                                            spec_mask=spec_label2)

        if idx == 0:
            dataset1 = numpy.zeros((vector_array1.shape[0] * len(file_list), dims), float)
            label1 = numpy.zeros((vector_array1.shape[0] * len(file_list), dims), float)
            
            dataset2 = numpy.zeros((vector_array2.shape[0] * len(file_list), dims), float)
            label2 = numpy.zeros((vector_array2.shape[0] * len(file_list), dims), float)
              
        dataset1[vector_array1.shape[0] * idx: vector_array1.shape[0] * (idx + 1), :] = vector_array1
        label1[vector_array1.shape[0] * idx: vector_array1.shape[0] * (idx + 1), :] = spec_label1
        
        dataset2[vector_array2.shape[0] * idx: vector_array2.shape[0] * (idx + 1), :] = vector_array2
        label2[vector_array2.shape[0] * idx: vector_array2.shape[0] * (idx + 1), :] = spec_label2

    return dataset1, dataset2, label1, label2


def train_list_to_mixture_spec_vector_array_fixed(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         target_source = None):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):

        sr, ys, src_file_lst = train_file_to_mixture_wav(file_list[idx])  # mixture
        
        target_idx = machine_types.index(target_source)
        
        _, target_sound = demux_wav(src_file_lst[target_idx])  
        _, label_spec = generate_label(numpy.expand_dims(target_sound, axis=0), MACHINE)
        spec_label = label_spec[:1, :, :].unsqueeze(3).repeat(1, 1, 1, n_mels).reshape(1, 309, frames * n_mels).squeeze(0).numpy()
     
        vector_array = wav_to_spec_vector_array(sr, ys,
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)

        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
            label = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
              
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
        label[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = spec_label
        
    return dataset, label  # mixture, target_label





class AEDatasetMix(torch.utils.data.Dataset):
    def __init__(self, 
            file_list,
            param,
            target_source=None,
            ):
        self.file_list = file_list
        self.target_source = target_source

        self.data_vector1, self.data_vector2, self.label1, self.label2 = train_list_to_mixture_spec_vector_array(self.file_list,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            )        
    
    def __getitem__(self, index):
        return torch.Tensor(self.data_vector1[index, :]), torch.Tensor(self.data_vector2[index, :]), torch.Tensor(self.label1[index, :]), torch.Tensor(self.label2[index, :])
    
    def __len__(self):
        return self.data_vector1.shape[0]

  
class AEDatasetMixFixed(torch.utils.data.Dataset):
    def __init__(self, 
            file_list,
            param,
            target_source=None,
            ):
        self.file_list = file_list
        self.target_source = target_source

        self.data_vector, self.label = train_list_to_mixture_spec_vector_array_fixed(self.file_list,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            target_source = self.target_source
                                            )        
    
    def __getitem__(self, index):
        return torch.Tensor(self.data_vector[index, :]), torch.Tensor(self.label[index, :])
    
    def __len__(self):
        return self.data_vector.shape[0]
      
      

def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 
    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """

    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext))))
    normal_len = [len(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir.replace(S1, mt),
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext)))) for mt in machine_types]
    normal_len = min(min(normal_len), len(normal_files))
    normal_files = normal_files[:normal_len]

    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = []
    for machine_type in machine_types:
        abnormal_files.extend(sorted(glob.glob(
            os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir.replace(S1, machine_type),
                                                                 abnormal_dir_name=abnormal_dir_name,
                                                                 ext=ext)))))         

    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[num_eval_normal:]
    train_labels = normal_labels[num_eval_normal:]
    
    ######################################################################################################################## 
    eval_normal_files = sum([[fan_file.replace(S1, machine_type) for fan_file in normal_files[:num_eval_normal]] for machine_type in machine_types], [])
    eval_files = numpy.concatenate((eval_normal_files, abnormal_files), axis=0)
    eval_labels = numpy.concatenate((np.repeat(normal_labels[:num_eval_normal], len(machine_types)), abnormal_labels), axis=0)  ##TODO 
    ######################################################################################################################## 
   
    # eval_files = numpy.concatenate((normal_files[:num_eval_normal], abnormal_files), axis=0)
    # eval_labels = numpy.concatenate((normal_labels[:num_eval_normal], abnormal_labels), axis=0)
    
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels


########################################################################
# main
########################################################################
if __name__ == "__main__":

    # load parameter yaml
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    # set random seed fixed
    fix_seed(param['seed'])

    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/6dB/{machine}/{source}".format(base=param["base_directory"], machine = MACHINE, source = S1))))  # {base}/0dB/fan/id_00/normal/00000000.wav

    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    # loop of the base directory
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

        # dataset param        
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = 'mix'
        machine_id = os.path.split(target_dir)[1]

        # setup path
        evaluation_result = {}
        train_pickle = "{pickle}/train_{machine_type}_{machine_id}_{db}.pickle".format(pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id, db=db)
        eval_files_pickle = "{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        eval_labels_pickle = "{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        model_file = "{model}/model_{machine_type}_{machine_id}_{db}.hdf5".format(model=param["model_directory"],
                                                                                  machine_type=machine_type,
                                                                                  machine_id=machine_id,
                                                                                  db=db)
        history_img = "{model}/history_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)
        evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(machine_type=machine_type,
                                                                          machine_id=machine_id,
                                                                          db=db)

        # dataset generator
        print("============== DATASET_GENERATOR ==============")
        train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)
        save_pickle(train_pickle, train_files)
        save_pickle(eval_files_pickle, eval_files)
        save_pickle(eval_labels_pickle, eval_labels)

        model = {}
        for target_type in machine_types:
            train_dataset = AEDatasetMixFixed(train_files, param, target_source=target_type)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=param["fit"]["batch_size"], shuffle=True,)

            # model training
            print("============== MODEL TRAINING ==============")
            dim_input = train_dataset.data_vector.shape[1]
            model[target_type] = TorchModel(dim_input).cuda()
            optimizer = torch.optim.Adam(model[target_type].parameters(), lr=1.0e-2)

            for epoch in range(param["fit"]["epochs"]):
                losses = []
                for batch, label in train_loader:
                    batch = batch.cuda()
                    label = label.cuda()
                    pred = model[target_type](batch * label)  # batch: mixture nomasked             
                                
                    loss = torch.mean(((pred - (batch * label))*label)**2)  
                
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                if epoch % 10 == 0:
                    print(f"epoch {epoch}: loss {sum(losses) / len(losses)}")
            model[target_type].eval()
               
        # evaluation
        print("============== EVALUATION ==============")
        y_pred_mean = numpy.array([0. for k in eval_labels])
        y_pred_mask = numpy.array([0. for k in eval_labels])
        y_true = numpy.array(eval_labels)

        eval_types = {mt: [] for mt in machine_types}
        overlap_ratios = []
                    
        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            machine_type = os.path.split(os.path.split(os.path.split(file_name)[0])[0])[1]
            target_idx = machine_types.index(machine_type)  
            
            sr, ys, y_raw, active_label_sources, active_spec_label_sources = eval_file_to_mixture_wav_label(file_name)
            overlap_ratios.append(get_overlap_ratio(active_label_sources[machine_types[0]], active_label_sources[machine_types[1]]))
            
            active_labels = torch.stack([active_label_sources[src] for src in machine_types])
           
            n_mels = param["feature"]["n_mels"]
            frames = param["feature"]["frames"]
            # [1, 309, 5] -> [309, 5*n_mels]
            active_spec_label = active_spec_label_sources[machine_type][:1, :, :].cuda().unsqueeze(3).repeat(1, 1, 1, n_mels).reshape(1, 309, frames * n_mels).squeeze(0)
            active_ratio = torch.sum(active_spec_label) / torch.sum(torch.ones_like(active_spec_label))

            data = wav_to_spec_vector_array(sr, ys,
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"],)
            

            data = torch.Tensor(data).cuda()
            error = torch.mean(((data - model[machine_type](data)) ** 2), dim=1)
            error_mask = torch.mean(((data * active_spec_label - model[machine_type](data * active_spec_label)) * active_spec_label) ** 2, dim=1)

            y_pred_mean[num] = torch.mean(error).detach().cpu().numpy()
            y_pred_mask[num] = (torch.mean(error_mask) / active_ratio).detach().cpu().numpy()
            
            eval_types[machine_type].append(num)

        mean_scores = []
        mask_scores = []
        anomaly_detect_score = {}

        for machine_type in machine_types:
            mean_score = metrics.roc_auc_score(y_true[eval_types[machine_type]], y_pred_mean[eval_types[machine_type]])
            mask_score = metrics.roc_auc_score(y_true[eval_types[machine_type]], y_pred_mask[eval_types[machine_type]])
            logger.info("AUC_mean_{} : {}".format(machine_type, mean_score))
            logger.info("AUC_mask_{} : {}".format(machine_type, mask_score))
            evaluation_result["AUC_mean_{}".format(machine_type)] = float(mean_score)
            evaluation_result["AUC_mask_{}".format(machine_type)] = float(mask_score)
            mean_scores.append(mean_score)
            mask_scores.append(mask_score)
        
        mean_score = sum(mean_scores) / len(mean_scores)
        mask_score = sum(mask_scores) / len(mask_scores)
        logger.info("AUC_mean : {}".format(mean_score))
        logger.info("AUC_mask : {}".format(mask_score))
        overall_overlap = sum(overlap_ratios) / len(overlap_ratios)
        logger.info("overlap_ratio : {}".format(overall_overlap))
        evaluation_result["AUC_mean"] = float(mean_score)
        evaluation_result["AUC_mask"] = float(mask_score)
        evaluation_result["overlap_ratio"] = float(overall_overlap)
        results[evaluation_result_key] = evaluation_result
        print("===========================")

    # output results
    print("\n===========================")
    logger.info("all results -> {}".format(result_file))
    with open(result_file, "w") as f:
        f.write(yaml.dump(results, default_flow_style=False))
    print("===========================")
########################################################################