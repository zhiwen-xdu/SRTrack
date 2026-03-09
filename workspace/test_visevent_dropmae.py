import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.siamtrack_dropmae import SIAMTrack
import lib.test.parameter.vipt_dropmae as rgbe_prompt_params
from lib.train.dataset.event_utils import get_merge_frame
import multiprocessing
import torch
import time

import sys
sys.path.append("/home/czwos/Project/SIAMTrack/lib/train")


def genConfig(seq_path, set_type):
    if set_type == 'VisEvent':
        rgb_img_list = sorted([seq_path + '/img/' + p for p in os.listdir(seq_path + '/img') if os.path.splitext(p)[1] == '.jpg'])
        event_img_list = sorted([seq_path + '/evimg/' + p for p in os.listdir(seq_path + '/evimg') if os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
        absent_label = np.loadtxt(seq_path + '/absent_label.txt')

    return rgb_img_list, event_img_list, gt, absent_label



def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=60, experiment_id="01", script_name='prompt'):
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    seq_txt = seq_name
    save_name = '{}/ep_{}'.format(experiment_id,epoch)
    save_folder = f'./workspace/results/{dataset_name}/' + save_name
    save_path = f'./workspace/results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return

    if script_name == 'vipt':
        params = rgbe_prompt_params.parameters(yaml_name,experiment_id,epoch)
        ostrack = SIAMTrack(params)
        tracker = SIAMTracker(tracker=ostrack)

    seq_path = seq_home + '/' + seq_name
    print('Process sequence: '+ seq_name)

    rgb_img_list, event_img_list, gt, absent_label = genConfig(seq_path, dataset_name)
    if absent_label[0] == 0: # first frame is absent in some seqs
        first_present_idx = absent_label.argmax()
        rgb_img_list = rgb_img_list[first_present_idx:]
        event_img_list = event_img_list[first_present_idx:]
        gt = gt[first_present_idx:]

    if len(rgb_img_list) == len(gt):
        result = np.zeros_like(gt)
    else:
        result = np.zeros((len(rgb_img_list), 4), dtype=gt.dtype)
    result[0] = np.copy(gt[0])
    toc = 0
    # try:
    for frame_idx, (rgb_path, event_path) in enumerate(zip(rgb_img_list, event_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            image = get_merge_frame(rgb_path, event_path)
            tracker.initialize(image, gt[0].tolist())  # xywh
        elif frame_idx > 0:
            image = get_merge_frame(rgb_path, event_path)
            region, confidence = tracker.track(image)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    np.savetxt(save_path, result, fmt='%.14f', delimiter=',')
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class SIAMTracker(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    parser = argparse.ArgumentParser(description='Run tracker on RGBE dataset.')
    parser.add_argument('--script_name', type=str, default='vipt', help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='droptrack_rgbx', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='VisEvent', help='Name of dataset (VisEvent).')
    parser.add_argument('--threads', default=32, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=8, type=int, help='Number of gpus')
    parser.add_argument('--experiment_id', default="DropTrack-RGBX", type=str, help='epochs of ckpt')
    parser.add_argument('--epoch', default=30, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='running mode: [sequential , parallel]')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', type=str, default='', help='Sequence name for debug.')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    cur_dir = abspath(dirname(__file__))
    # path initialization
    seq_list = None
    if dataset_name == 'VisEvent':
        seq_home = '/home/czwos/Data/VisEvent'
        with open('/home/czwos/Project/SIAMTrack/lib/train/data_specs/visevent_val_split.txt', 'r') as f:
            seq_list = f.read().splitlines()
        seq_list.sort()
    else:
        raise ValueError("Error dataset!")

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.experiment_id, args.script_name) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.experiment_id, args.script_name) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)

    print(f"Totally cost {time.time()-start} seconds!")


