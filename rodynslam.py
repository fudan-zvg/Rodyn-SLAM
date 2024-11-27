import os
#os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from model.lsd import LSDDetector
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image, get_distance_map, warp_flow, compute_fwdbwd_mask, get_uv_grid, compute_sampson_error
# from tools.eval_ate_align import pose_evaluation
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion

# load optical flow
from model.RAFT.raft import RAFT
from model.RAFT.utils import flow_viz
from model.RAFT.utils.utils import InputPadder
from model.GMA.network import RAFTGMA

def get_flow_model(config):
    if config["flow"]["flow_model"] == 'raft':
        flow_model = RAFT
    elif config["flow"]["flow_model"] == 'gma':
        flow_model = RAFTGMA
    else:
        flow_model = RAFT

    return flow_model()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class RodynSLAM():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        
        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)

        self.model = JointEncoding(config, self.bounding_box).to(self.device)
        self.sample_uv = []
        self.filter_sample_mask = None
        self.line_detector = LSDDetector()
        self.use_line_feature = self.config['training']['line_feature']

        self.flow_model = torch.nn.DataParallel(get_flow_model(config))
        self.flow_model.load_state_dict(torch.load(config["flow"]["checkpoint"]))
        self.flow_model = self.flow_model.module
        self.flow_model.to('cuda')
        self.flow_model.eval()
        
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')
        
        elif self.config['training']['rot_rep'] == "quat":
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
            print("Using quaternion as rotation representation")
        else:
            raise NotImplementedError
        
    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.load_gt_pose() 
    
    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(self.device)

    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.dataset.H, 
                                self.dataset.W, 
                                num_kf, 
                                self.dataset.num_rays_to_save, 
                                self.device)
    
    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose
 
    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
    
    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']

    def sampling_without_replacement(self, logp, k):
        def gumbel_like(u):
            return -torch.log(-torch.log(torch.rand_like(u) + 1e-7) + 1e-7)

        scores = logp + gumbel_like(logp)
        return scores.topk(k, dim=-1)[1]

    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice

    def select_samples_with_mask(self, iH, iW, H, W, depth, samples, seg_mask, dist_map, motion_mask=None, device="cuda:0"):
        '''
        randomly select samples from the image with mask
        '''
        if motion_mask is None:
            seg_mask = seg_mask[:, iH:H-iH, iW:W-iW]
            seg_mask = seg_mask.to(device)
            mask = 1-seg_mask
        else:
            motion_mask = motion_mask[:, iH:H - iH, iW:W - iW]
            motion_mask = motion_mask.to(device)
            mask = 1 - motion_mask
        depth = depth[:, iH:H - iH, iW:W - iW]
        valid_depth = torch.where(depth > 0, torch.ones_like(depth), torch.zeros_like(depth))
        valid_depth = valid_depth.to(device)

        dist_map = dist_map.unsqueeze(0).to(device)
        dist_map = dist_map[:, iH:H - iH, iW:W - iW]

        dist_mask = torch.where(dist_map > 6.0, torch.ones_like(dist_map), torch.zeros_like(dist_map))
        mask = mask * valid_depth * dist_mask

        mask_valid = torch.nonzero(mask.squeeze())
        num_valid = len(mask_valid)
        idxs = random.sample(range(0, num_valid), samples)
        sampled_index = mask_valid[idxs, :]

        return sampled_index

    def sample_whole_img(self, H, W, device="cuda:0"):
        img = torch.ones(H, W)
        img = img.to(device)
        sampled_index = torch.nonzero(img)
        return sampled_index

    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]

        if smooth and self.config['training']['smooth_weight']>0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'], 
                                                                                  self.config['training']['smooth_vox'], 
                                                                                  margin=self.config['training']['smooth_margin'])

        return loss

    def ste_round(self, x):
        return torch.round(x) - x.detach() + x
    def compute_edge_dt_loss(self, uv, dt_map, loss_type='huber'):
        u = torch.round(uv[:, 1]).long()
        v = torch.round(uv[:, 0]).long()
        loss_map_old = dt_map.squeeze()[u, v]
        # # uv normalize
        uv[..., 0] = uv[..., 0] / dt_map.shape[2] * 2.0 - 1.0
        uv[..., 1] = uv[..., 1] / dt_map.shape[1] * 2.0 - 1.0

        # grid sample strategy 2
        uv = uv[None, :, None, :]
        dt_map = dt_map[None, ...].to(self.device)
        loss_map = torch.nn.functional.grid_sample(dt_map, uv, padding_mode="border", align_corners=True)
        loss_map = loss_map.squeeze()

        loss_map_mask = loss_map < 10
        loss_map_valid = loss_map[loss_map_mask]
        gt_loss_map = torch.zeros_like(loss_map_valid)

        if loss_type == 'l2':
            return F.mse_loss(loss_map_valid, gt_loss_map)
        elif loss_type == 'l1':
            return F.l1_loss(loss_map_valid, gt_loss_map)
        elif loss_type == 'huber':
            return F.huber_loss(loss_map_valid, gt_loss_map, delta=0.3)
        raise Exception('Unsupported loss type')

    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        print('First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w.clone()
        self.est_c2w_data_rel[0] = torch.from_numpy(np.eye(4)).float().to(self.device)

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples_with_mask(0, 0, self.dataset.H, self.dataset.W, batch['depth'],
                                                   self.config['mapping']['sample'], batch['seg_mask'], batch['seg_dist_map'], device=self.device)
            indice_h, indice_w = indice[:, 0], indice[:, 1]
            self.sample_uv = indice

            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            rays_edge_dist = batch['seg_dist_map'][indice_h, indice_w].to(self.device).unsqueeze(-1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d, ray_dist=rays_edge_dist)
            self.filter_sample_mask = ret["valid_depth_mask"]
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()
        
        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        if self.config['mapping']['first_mesh']:
            self.save_mesh(0)
        
        print('First frame mapping done')
        return ret, loss

    def current_frame_mapping(self, batch, cur_frame_id):
        '''
        Current frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float

        '''
        if self.config['mapping']['cur_frame_iters'] <= 0:
            return
        print('Current frame mapping...')

        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()

        # Training
        for i in range(self.config['mapping']['cur_frame_iters']):
            self.cur_map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])

            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.cur_map_optimizer.step()

        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss

    def freeze_model(self):
        '''
        Freeze the model parameters
        '''
        for param in self.model.embed_fn.parameters():
            param.require_grad = False
        
        for param in self.model.decoder.parameters():
            param.require_grad = False

    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                               {"params": cur_trans, "lr": self.config[task]['lr_trans']}])
        
        return cur_rot, cur_trans, pose_optimizer

    def get_edge_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot_edge']},
                                           {"params": cur_trans, "lr": self.config[task]['lr_trans_edge']}])

        return cur_rot, cur_trans, pose_optimizer

    def global_BA(self, batch, cur_frame_id):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        
        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]

            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
        
        # Set up optimizer include mapping
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.squeeze()

        for i in range(self.config['mapping']['iters']):
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])
            idx_cur = self.select_samples_with_mask(0, 0, self.dataset.H, self.dataset.W, batch['depth'],
                                                    max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']), batch['seg_mask'], batch['seg_dist_map'], motion_mask=batch['motion_mask'], device=self.device)

            current_rays_batch = current_rays[idx_cur[:, 0], idx_cur[:, 1]]

            rays_edge_dist = batch['seg_dist_map'][idx_cur[:, 0], idx_cur[:, 1]].to(self.device).unsqueeze(-1)
            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)


            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret, smooth=True)

            loss.backward(retain_graph=True)
            
            if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:
               
                if (i + 1) > cfg["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses
                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)

                # zero_grad here
                pose_optimizer.zero_grad()
        
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]

    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev.clone()
        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id-2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            delta = c2w_est_prev@c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta@c2w_est_prev
        
        return self.est_c2w_data[frame_id]

    def tracking_pc(self, batch, frame_id):
        '''
        Tracking camera pose of current frame using point cloud loss
        (Not used in the paper, but might be useful for some cases)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)

        cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        indice = None
        cur_trans = torch.nn.parameter.Parameter(cur_c2w[..., :3, 3].unsqueeze(0))
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(cur_c2w[..., :3, :3]).unsqueeze(0))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config['tracking']['lr_rot']},
                                               {"params": cur_trans, "lr": self.config['tracking']['lr_trans']}])
        best_sdf_loss = None

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        thresh=0

        if self.config['tracking']['iter_point'] > 0:
            indice_pc = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['pc_samples'])
            rays_d_cam = batch['direction'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_s = batch['rgb'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_d = batch['depth'][:, iH:-iH, iW:-iW].reshape(-1, 1)[indice_pc].to(self.device)

            valid_depth_mask = ((target_d > 0.) * (target_d < 5.))[:,0]

            rays_d_cam = rays_d_cam[valid_depth_mask]
            target_s = target_s[valid_depth_mask]
            target_d = target_d[valid_depth_mask]

            for i in range(self.config['tracking']['iter_point']):
                pose_optimizer.zero_grad()
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                rays_o = c2w_est[...,:3, -1].repeat(len(rays_d_cam), 1)
                rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)
                pts = rays_o + target_d * rays_d

                pts_flat = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

                out = self.model.query_color_sdf(pts_flat)

                sdf = out[:, -1]
                rgb = torch.sigmoid(out[:,:3])
                loss = 5 * torch.mean(torch.square(rgb-target_s)) + 1000 * torch.mean(torch.square(sdf))

                if best_sdf_loss is None:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()

                with torch.no_grad():
                    c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                    if loss.cpu().item() < best_sdf_loss:
                        best_sdf_loss = loss.cpu().item()
                        best_c2w_est = c2w_est.detach()
                        thresh = 0
                    else:
                        thresh +=1
                if thresh >self.config['tracking']['wait_iters']:
                    print("thresh: ", thresh)
                    break

                loss.backward()
                pose_optimizer.step()
        

        if self.config['tracking']['best']:
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]


        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            # Not a keyframe, need relative pose
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta
        print('Best loss: {}, Camera loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))

    def get_edge_opt_pose(self, cur_c2w, last_kf_c2w, batch, last_kf_batch):
        best_edge_loss = None
        thresh = 0

        cur_rot, cur_trans, pose_optimizer = self.get_edge_pose_param_optim(cur_c2w[None, ...], mapping=False)

        last_seg_dt_map = last_kf_batch['seg_dist_map']
        last_seg_dt_map = last_seg_dt_map.unsqueeze(0).to(self.device)
        last_seg_dt_mask = last_seg_dt_map > 6.0

        # check pass
        last_edge_map = last_kf_batch['edge']
        last_edge_map[~last_seg_dt_mask] = 0
        edge_uv = torch.nonzero(last_edge_map.squeeze())

        # OpenGL coordinate
        edge_u = edge_uv[:, 0]
        edge_v = edge_uv[:, 1]

        rays_d_cam = last_kf_batch['direction'].squeeze(0)[edge_u, edge_v, :].to(self.device)
        rays_o = last_kf_c2w[None, :3, -1].repeat(edge_uv.shape[0], 1)
        last_kf_c2w = last_kf_c2w.unsqueeze(0)
        rays_d = torch.sum(rays_d_cam[..., None, :] * last_kf_c2w[:, :3, :3], -1)
        edge_depth = last_kf_batch['depth'].squeeze(0)[edge_u, edge_v][:, None].to(rays_o)

        pts3d_w = (rays_o + rays_d * edge_depth).float()

        # Start tracking
        for i in range(self.config['tracking']['edge_iter']):
            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)
            w2c_est = torch.inverse(c2w_est)
            pts3d_homo = torch.cat([pts3d_w, torch.ones((pts3d_w.shape[0], 1)).to(pts3d_w)], dim=1)
            pts2d_homo = w2c_est @ pts3d_homo[:, None, :, None]  # [Cn, 4, 4] @ [Pn, 1, 4, 1] = [Pn, Cn, 4, 1]
            pts2d = pts2d_homo[:, :, :3]
            K = batch["k_matrix"][:3, :3].to(pts2d).squeeze()
            pts2d[:, :, 0] *= -1
            uv = K @ pts2d  # [3,3] @ [Pn, Cn, 3, 1] = [Pn, Cn, 3, 1]
            z = uv[:, :, -1:] + 1e-5
            uv = uv[:, :, :2] / z  # [Pn, Cn, 2, 1]
            uv = uv.float()
            uv = uv.view(pts3d_w.shape[0], 2)
            z = z.view(pts3d_w.shape[0], 1)

            # filter mask: bound mask + z valid mask + cur seg mask
            bound_mask = (torch.round(uv[:, 1]).long() < batch['edge'].shape[1]) * (torch.round(uv[:, 1]).long() >= 0) \
                         * (torch.round(uv[:, 0]).long() < batch['edge'].shape[2]) * (torch.round(uv[:, 0]).long() > 0)

            # use z < 0 filter
            valid_mask = bound_mask & (z.squeeze() <= 0)
            valid_uv = uv[valid_mask]
            cur_seg_dt_map = batch['seg_dist_map']
            cur_seg_dt_map = cur_seg_dt_map.unsqueeze(0).to(self.device)

            cur_seg_dt_mask = cur_seg_dt_map > 6.0
            rpj_u = torch.round(valid_uv[:, 1]).long()
            rpj_v = torch.round(valid_uv[:, 0]).long()
            cur_seg_mask = cur_seg_dt_mask.squeeze()[rpj_u, rpj_v]
            filter_uv = valid_uv[cur_seg_mask]

            # run optimizer
            loss = self.compute_edge_dt_loss(filter_uv, batch['edge_dt'])
            # print('Iter edge loss: {}'.format(loss.cpu().item()))

            if best_edge_loss is None:
                best_edge_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                if loss.cpu().item() < best_edge_loss:
                    best_edge_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh +=1

            loss.backward()
            pose_optimizer.step()

        return best_c2w_est.detach().clone()[0]

    def refine_motion_mask(self, cur_batch, cur_frame_id):
        flow_img_batch = []
        window_size = self.config['flow']['window_size']
        with torch.no_grad():
            if len(self.keyframeDatabase.frame_ids) <= window_size:
                for kf_batch in self.keyframeDatabase.keyframes.values():
                    kf_img = kf_batch["rgb"].squeeze() * 255.0
                    kf_img = kf_img.permute(2, 0, 1)
                    flow_img_batch.append(kf_img)
            else:
                flow_img_batch = [list(self.keyframeDatabase.keyframes.values())[i]["rgb"].squeeze().permute(2, 0, 1) * 255.0 for i in range(-window_size, 0)]

            kf_img_batch = torch.stack(flow_img_batch, dim=0).to(self.device)
            cur_kf_img = cur_batch["rgb"].squeeze().permute(2, 0, 1) * 255.0
            cur_img_batch = cur_kf_img.repeat(kf_img_batch.shape[0], 1, 1, 1).to(self.device)
            padder = InputPadder(kf_img_batch.shape)
            his_imgs, cur_imgs = padder.pad(kf_img_batch, cur_img_batch)
            _, flow_bwd = self.flow_model(cur_imgs, his_imgs, iters=30, test_mode=True)

            flow_bwd = padder.unpad(flow_bwd).permute(0, 2, 3, 1)

            H = cur_img_batch.shape[2]
            W = cur_img_batch.shape[3]
            uv = get_uv_grid(H, W, align_corners=False)
            x1 = uv.reshape(-1, 2)
            flow_bwd_norm = torch.stack([2.0 * flow_bwd[..., 0] / (W - 1), 2.0 * flow_bwd[..., 1] / (H - 1)], axis=-1)
            err_batch = []
            for i in range(flow_bwd_norm.shape[0]):
                flow_tmp = flow_bwd_norm[i].cpu()
                x2 = x1 + flow_tmp.view(-1, 2)
                F, mask = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
                F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
                err = compute_sampson_error(x1, x2, F).reshape(H, W)
                fac = (H + W) / 2
                err = err * fac ** 2
                err_batch.append(err)

            error_batch = torch.stack(err_batch, 0)
            thresh = torch.quantile(error_batch.view(len(err_batch), -1), 0.85, dim=-1)
            thresh = thresh[:, None, None].repeat(1, H, W)
            err_map = torch.where(error_batch <= thresh, torch.zeros_like(error_batch), torch.ones_like(error_batch))

            finial_error_map = torch.ones_like(err_map[0])
            for j in range(err_map.shape[0]):
                finial_error_map *= err_map[j]

            seg_mask_static = cur_batch["seg_mask"].squeeze()

            finial_motion_map = finial_error_map.int() | seg_mask_static
            return finial_motion_map

    def tracking_render(self, batch, frame_id):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''
        c2w_gt = batch['c2w'][0].to(self.device)

        if self.config['tracking']['iter_point'] > 0:
            cur_c2w = self.est_c2w_data[frame_id]
        else:
            cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        last_kf_batch = list(self.keyframeDatabase.keyframes.values())[-1]
        last_kf_id = list(self.keyframeDatabase.keyframes.keys())[-1].item()
        if last_kf_batch is not None:
            last_kf_c2w = self.est_c2w_data[last_kf_id]
            cur_c2w_edge = self.get_edge_opt_pose(cur_c2w, last_kf_c2w, batch, last_kf_batch)
            cur_c2w = cur_c2w_edge.detach().clone()

        indice = None
        best_sdf_loss = None
        thresh=0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None,...], mapping=False)

        # Start tracking
        for i in range(self.config['tracking']['iter']):
            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

            # Note here we fix the sampled points for optimisation
            if indice is None:
                indice = self.select_samples_with_mask(iH, iW, self.dataset.H, self.dataset.W, batch['depth'], self.config['tracking']['sample'],
                                              batch['seg_mask'], batch['seg_dist_map'], motion_mask=batch['motion_mask'], device=self.device)

                # Slicing
                indice_h, indice_w = indice[:, 0], indice[:, 1]
                self.sample_uv = indice
                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)

            target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_edge_dist = batch['seg_dist_map'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w_est[...,:3, -1].repeat(self.config['tracking']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d, ray_dist=rays_edge_dist)
            loss = self.get_loss_from_ret(ret)

            self.filter_sample_mask = ret["valid_depth_mask"]
            
            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh +=1
            
            if thresh > self.config['tracking']['wait_iters']:
                print("thresh: ", thresh)
                break

            loss.backward()
            pose_optimizer.step()

        if self.config['tracking']['best']:
            # Use the pose with smallest loss
            # use gt keyframe pose to verify edge warp loss
            if frame_id % self.config['mapping']['keyframe_every'] == 0:
                self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
            else:
                with torch.no_grad():
                    best_c2w_rays_o = best_c2w_est[..., :3, -1].repeat(self.config['tracking']['sample'], 1)
                    best_c2w_rays_d = torch.sum(rays_d_cam[..., None, :] * best_c2w_est[:, :3, :3], -1)

                    best_c2w_ret = self.model.forward(best_c2w_rays_o, best_c2w_rays_d, target_s, target_d, ray_dist=rays_edge_dist)
                    best_c2w_loss = self.get_loss_from_ret(best_c2w_ret)

                    edge_c2w_rays_o = cur_c2w_edge.unsqueeze(0)[..., :3, -1].repeat(self.config['tracking']['sample'], 1)
                    edge_c2w_rays_d = torch.sum(rays_d_cam[..., None, :] * cur_c2w_edge.unsqueeze(0)[:, :3, :3], -1)

                    edge_c2w_ret = self.model.forward(edge_c2w_rays_o, edge_c2w_rays_d, target_s, target_d, ray_dist=rays_edge_dist)
                    edge_c2w_loss = self.get_loss_from_ret(edge_c2w_ret)

                if (edge_c2w_loss < best_c2w_loss):
                    self.est_c2w_data[frame_id] = cur_c2w_edge.detach().clone()
                else:
                    self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            # Use the pose after the last iteration
            # self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]
            self.est_c2w_data[frame_id] = cur_c2w_edge.detach().clone()

       # Save relative pose of non-keyframes
        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta

        print('Best loss: {}, Last loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))
    
    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i]
                poses[i] = delta @ c2w_key
        
        return poses

    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
    
        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})

        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
    
    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'mesh_track{}.ply'.format(i))
        extract_mesh(self.model.query_sdf, 
                        self.config, 
                        self.bounding_box, 
                        color_func=self.model.query_color, 
                        marching_cube_bound=self.marching_cube_bound, 
                        voxel_size=voxel_size, 
                        mesh_savepath=mesh_savepath)
        
    def run(self):
        self.create_optimizer()
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])

        # Start RoDyn-SLAM!
        for i, batch in tqdm(enumerate(data_loader)):
            # Compute DT distance
            distance_map = get_distance_map(batch['seg_mask'])
            distance_map = torch.from_numpy(distance_map)
            batch['seg_dist_map'] = distance_map

            # extract line feature && compute line reprojection map
            # global BA may be difficult due to history overlap
            if self.use_line_feature:
                color_data = batch['rgb'].detach().cpu().numpy().squeeze(0)
                line_feature = self.line_detector.detect_with_min_length(color_data, 35)

            # First frame mapping
            if i == 0:
                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
                batch['motion_mask'] = torch.ones_like(batch['seg_mask'])

            # Tracking + Mapping
            else:
                # add flow + seg mask refine
                if self.config['training']['motion_mask']:
                    motion_mask = self.refine_motion_mask(batch, i)
                    batch['motion_mask'] = motion_mask.unsqueeze(0)
                else:
                    batch['motion_mask'] = None
                # Tracking thread
                if self.config['tracking']['iter_point'] > 0:
                    self.tracking_pc(batch, i)
                self.tracking_render(batch, i)

                # global ba finish mapping
                if i% self.config['mapping']['map_every'] == 0:
                    self.global_BA(batch, i)

                if i % self.config['mapping']['keyframe_every'] == 0:
                    self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
                    print('add keyframe:', i)

            # Visualisation
            if self.config['mesh']['visualisation']:
                rgb = cv2.cvtColor(batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB)
                raw_depth = batch["depth"]
                # 20230520 filter invalid depth
                mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
                depth_colormap = colormap_image(batch["depth"])
                depth_colormap[:, mask] = 255.
                depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
                
                seg_mask = batch["seg_mask"]
                seg_mask_colormap = colormap_image(seg_mask)
                seg_mask_colormap = seg_mask_colormap.permute(1, 2, 0).cpu().numpy()

                seg_mask_static = 1-seg_mask
                seg_mask_static_colormap = colormap_image(seg_mask_static)
                seg_mask_static_colormap = seg_mask_static_colormap.permute(1, 2, 0).cpu().numpy()

                rgb_sample = rgb.copy()
                aW = self.config['tracking']['ignore_edge_W']
                aH = self.config['tracking']['ignore_edge_H']
                filter_uv = self.sample_uv[self.filter_sample_mask]
                for uv in self.sample_uv:
                    if i == 0 :
                        cv2.circle(rgb_sample, (uv[1].item(), uv[0].item()), 3, (0, 0, 255), cv2.FILLED)
                    else:
                        cv2.circle(rgb_sample, (uv[1].item() + aW, uv[0].item() + aH), 3, (0, 0, 255), cv2.FILLED)

                    if uv not in filter_uv:
                        if i == 0:
                            cv2.circle(rgb_sample, (uv[1].item(), uv[0].item()), 3, (0, 0, 0), cv2.FILLED)
                        else:
                            cv2.circle(rgb_sample, (uv[1].item() + aW, uv[0].item() + aH), 3, (0, 0, 0), cv2.FILLED)

                mask_cat = np.hstack((seg_mask_static_colormap, rgb_sample))
                image = np.hstack((rgb, depth_colormap))
                image = np.vstack((image, mask_cat))
                cv2.namedWindow('RGB-D'.format(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RGB-D'.format(i), image)
                key = cv2.waitKey(1)

        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'checkpoint{}.pt'.format(i)) 
        
        self.save_ckpt(model_savepath)
        self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])

        pose_relative = self.convert_relative_pose()
        pose_evaluation(self.pose_gt, self.est_c2w_data, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)
        pose_evaluation(self.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')


if __name__ == '__main__':
    seed_everything(10)
    print('Start Running RoDyn-SLAM....')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy("rodynslam.py", os.path.join(save_path, 'rodynslam.py'))

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = RodynSLAM(cfg)

    slam.run()
