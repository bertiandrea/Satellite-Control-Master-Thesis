# satellite.py

from code.utils.satellite_util import quat_from_euler_xyz, sample_random_quaternion_batch, quat_diff, quat_diff_rad, quat_axis, quat_mul, quat_conjugate
from code.envs.vec_task import DRVecTask
from code.rewards.satellite_reward import REWARD_MAP

import isaacgym #BugFix
import torch
from isaacgym import gymutil, gymtorch, gymapi

from pathlib import Path
import numpy as np
import pandas as pd
import math
import os

from torch.utils.tensorboard import SummaryWriter

BASE_COLORS_SAT  = torch.tensor([[1,0,1], [0,1,1], [1,1,0]], dtype=torch.float)
BASE_COLORS_GOAL = torch.tensor([[0,0,1], [0,1,0], [1,0,0]], dtype=torch.float)

class Satellite(DRVecTask):
    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, is_eval):

        self.is_eval = is_eval

        self.dt =                    config["sim"]["dt"]
        self.max_episode_length =    int(config["env"]["max_episode_length"] / self.dt)

        self.env_spacing =           config["env"]["env_spacing"]

        self.asset_name =            config["env"]["asset"]["asset_name"]
        self.asset_root =            config["env"]["asset"]["asset_root"]
        self.asset_file =            config["env"]["asset"]["asset_file_name"]

        self.torque_scale =          config["env"]["torque_scale"]
        self.debug_arrows =          config["env"]["debug_arrows"]
        self.debug_prints =          config["env"]["debug_prints"]

        self.reward_fn = REWARD_MAP[config["reward"]["reward_function"]](
            **config["reward"][config["reward"]["reward_function"]]
        )

        ###################################################
        self.log_status =               config["log_status"]["log"]
        if self.log_status:
            self.log_status_interval =  config["log_status"]["log_interval"]
            self.writer =               SummaryWriter(log_dir = config["log_status"]["log_dir"])
        ###################################################

        ###################################################    
        if self.is_eval:
            self.log_trajectories =                     config["log_trajectories"]["log"]
            if self.log_trajectories:
                self.log_trajectories_file =            config["log_trajectories"]["log_file"]
                self.log_trajectories_interval =        config["log_trajectories"]["log_interval"]
                self.log_trajectories_flush_interval =  config["log_trajectories"]["log_flush"]
            
            self.explosion =            config["explosion"]["enabled"]
            if self.explosion:
                self.explosion_time =   int(config["explosion"]["explosion_time"] / self.dt)
                self.explosion_mean =   config["explosion"]["explosion_mean"]
                self.explosion_std =    config["explosion"]["explosion_std"]
            
            self.discretize_starting_pos =  config["env"]["discretize_starting_pos"]
        ###################################################    

        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)

        ################# SETUP SIM #################
        self.actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.actor_root_state).view(self.num_envs, 13)
        self.satellite_pos     = self.root_states[:, 0:3]
        self.satellite_quats   = self.root_states[:, 3:7]
        self.satellite_linvels = self.root_states[:, 7:10]
        self.satellite_angvels = self.root_states[:, 10:13]
        #############################################

        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.initial_root_states = self.root_states.detach().clone()
        print(f"Initial root states: {self.initial_root_states[0]}")
        ########################################

        self.prev_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.delta_actions = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.torque_tensor = torch.zeros((self.num_bodies * self.num_envs, 3), device=self.device)
        self.force_tensor = torch.zeros((self.num_bodies * self.num_envs, 3), device=self.device)
        self.root_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.int) * self.num_bodies

        ###################################################
        if self.is_eval and self.explosion:
            self.impulse = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        ###################################################

        ###################################################    
        if self.is_eval and self.log_trajectories:
                os.makedirs(os.path.dirname(self.log_trajectories_file), exist_ok=True)
                self.log_buffer = []
        ###################################################

    def create_sim(self) -> None:
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params) # Acquires the sim pointer
        self.create_envs(self.env_spacing, int(np.sqrt(self.num_envs)))
        ###################################################
        if self.randomize:
            print("Applying randomizations...")
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int)
            self.apply_randomizations(ids, self.dr_params)
        ###################################################

    def create_envs(self, spacing, num_per_row: int) -> None:
        self.asset = self.load_asset()
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.actor_handles = []
        self.sat_glob_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        ###################################################
        if self.is_eval and self.discretize_starting_pos:
            print("[EVAL] Discretized starting position enabled. Sampling base quaternion.")
            base_quat = sample_random_quaternion_batch(self.device, 1)
            self.goal_quat = base_quat.repeat(self.num_envs, 1)
        else:
            self.goal_quat = sample_random_quaternion_batch(self.device, self.num_envs)

        if self.is_eval and self.discretize_starting_pos:
            print("[EVAL] Discretized starting position enabled. Precomputing orientations.")
            self.asset_init_pos_r_all = self.get_discretized_orientations(base_quat)
        ###################################################
            
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            origin = self.gym.get_env_origin(env)
            self.sat_glob_pos[i] = torch.tensor([origin.x, origin.y, origin.z],
                                                dtype=torch.float,
                                                device=self.device)
            ###################################################
            asset_init_pos_p = [0, 0, 0]

            if self.is_eval and self.discretize_starting_pos:
                #self.asset_init_pos_r_all = self.get_discretized_orientations(base_quat)
                asset_init_pos_r = self.asset_init_pos_r_all[i].cpu().numpy()
            else:
                asset_init_pos_r = sample_random_quaternion_batch(self.device, 1)[0].cpu().numpy()
            ###################################################
            actor_handle = self.create_actor(i, env, self.asset, asset_init_pos_p, asset_init_pos_r, 1, self.asset_name)
            ###################################################
            self.actor_handles.append(actor_handle)
            self.envs.append(env)

    def load_asset(self):
        asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file)
        self.num_bodies = self.gym.get_asset_rigid_body_count(asset)
        return asset
    
    def create_actor(self, env_idx: int, env, asset_handle, pose_p, pose_r, collision: int, name: str) -> None:
        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(*pose_p)
        init_pose.r = gymapi.Quat(*pose_r)
        actor_handle = self.gym.create_actor(env, asset_handle, init_pose, f"{name}", env_idx, collision)
        return actor_handle

    ################################################################################################################################

    def draw_arrows(self):
        x_goal = quat_axis(self.goal_quat, 0)
        y_goal = quat_axis(self.goal_quat, 1)
        z_goal = quat_axis(self.goal_quat, 2)
        x_sat  = quat_axis(self.satellite_quats, 0)
        y_sat  = quat_axis(self.satellite_quats, 1)
        z_sat  = quat_axis(self.satellite_quats, 2)

        sat_lines = torch.cat([
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + x_sat * 1.5], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + y_sat * 1.5], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + z_sat * 1.5], dim=1),
        ], dim=0)  # → (3N,2,3)
        goal_lines = torch.cat([
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + x_goal * 2.0], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + y_goal * 2.0], dim=1),
            torch.stack([self.sat_glob_pos, self.sat_glob_pos + z_goal * 2.0], dim=1),
        ], dim=0)  # → (3N,2,3)
        all_lines = torch.cat([sat_lines, goal_lines], dim=0)  # → (6N,2,3)

        colors_sat  = BASE_COLORS_SAT.repeat_interleave(self.num_envs, dim=0)   # (3N,3)
        colors_goal = BASE_COLORS_GOAL.repeat_interleave(self.num_envs, dim=0)  # (3N,3)
        all_colors = torch.cat([colors_sat, colors_goal], dim=0)  # (6N,3)

        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(
            self.viewer,
            None,
            6 * self.num_envs,
            all_lines.cpu().numpy(),
            all_colors.cpu().numpy()
        )

    ################################################################################################################################
    def get_discretized_orientations(self, base_quat: torch.Tensor) -> torch.Tensor:
        if self.num_envs == 1:
            return quat_conjugate(base_quat)

        pts = int(round(self.num_envs ** (1/3)))
        if pts ** 3 != self.num_envs:
            raise ValueError("num_envs must be a perfect cube for discretized orientations.")

        angles = torch.linspace(0, 2 * torch.pi, pts, device=self.device)
        angles_i, angles_j, angles_k = torch.meshgrid(angles, angles, angles, indexing='ij')
        orientations = quat_from_euler_xyz(angles_i.flatten(), angles_j.flatten(), angles_k.flatten())
        orientations = quat_mul(base_quat.repeat(self.num_envs, 1), orientations)
        orientations = orientations / torch.norm(orientations, dim=1, keepdim=True)
        return orientations
   
    ################################################################################################################################

    def explosion_impulse(self) -> torch.Tensor:
        mag = torch.normal(self.explosion_mean, self.explosion_std, size=(self.num_envs, 1), device=self.device)
        dirs = torch.randn(self.num_envs, 3, device=self.device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        impulse = dirs * mag

        impulse_x, impulse_y, impulse_z = impulse[0].tolist()
        print(("!" * 80 + "\n") * 15)
        print(f"!!!!!!!!!!!!!!!!!!!!!!! EXPLOSION {impulse_x:.2f} {impulse_y:.2f} {impulse_z:.2f} !!!!!!!!!!!!!!!!!!!!!!!\n")
        print(("!" * 80 + "\n") * 15)

        return impulse

    ################################################################################################################################
    
    def _log_scalar(self, tag: str, value: float):
        if self.control_steps % self.log_status_interval == 0:
            self.writer.add_scalar(tag, value, global_step=self.control_steps)
            self.writer.flush()

    def log_status_func(self) -> None:
        #################################################################
        q_diff = quat_diff(self.satellite_quats, self.goal_quat)
        q_diff_rad = quat_diff_rad(self.satellite_quats, self.goal_quat)
        goal = torch.lt(q_diff_rad * 180.0 / torch.pi, 0.1732).sum(dim=0)
        #################################################################

        self._log_scalar("Actions/action_X", self.actions[0, 0].item())
        self._log_scalar("Actions/action_Y", self.actions[0, 1].item())
        self._log_scalar("Actions/action_Z", self.actions[0, 2].item())

        self._log_scalar("Angular Error/q_diff_0", q_diff[0, 0].item())
        self._log_scalar("Angular Error/q_diff_1", q_diff[0, 1].item())
        self._log_scalar("Angular Error/q_diff_2", q_diff[0, 2].item())
        self._log_scalar("Angular Error/q_diff_3", q_diff[0, 3].item())
        self._log_scalar("Angular Error/q_diff_rad", q_diff_rad[0].item())
        self._log_scalar("Angular Error/q_diff_deg", q_diff_rad[0].item() * 180.0 / torch.pi)

        self._log_scalar("Angular Error/q_diff_mean_rad", q_diff_rad.mean().item())
        self._log_scalar("Angular Error/q_diff_mean_deg", q_diff_rad.mean().item() * 180.0 / torch.pi)

        self._log_scalar("Energy/mean", (self.actions ** 2).sum(dim=-1).mean().item())
        self._log_scalar("Energy/delta_mean", (self.delta_actions ** 2).sum(dim=-1).mean().item())

        self._log_scalar("Torque/max_mean", self.actions.abs().max(dim=1).values.mean().item())

        self._log_scalar('Goal/goal', goal.item())

    def log_trajectories_func(self) -> None:
        if self.control_steps % self.log_trajectories_interval == 0:
            self.log_buffer.append({
                "step": int(self.control_steps),
                "quat": self.satellite_quats.detach().cpu(),
                "ang_diff": quat_diff_rad(self.satellite_quats, self.goal_quat).detach().cpu() * (180.0 / math.pi),
                "angvel": self.satellite_angvels.detach().cpu(),
                "angacc": self.satellite_angacc.detach().cpu(),
                "actions": self.actions.detach().cpu(),
                "delta_actions": self.delta_actions.detach().cpu(),
            })

        if self.log_buffer and (
                    self.control_steps % self.log_trajectories_flush_interval == 0 or 
                    self.reset_buf.any().item()
                ):
            tmp_path = f"{self.log_trajectories_file}.tmp"
            if os.path.exists(self.log_trajectories_file):
                data = torch.load(self.log_trajectories_file, weights_only=True)
                data.extend(self.log_buffer)
            else:
                data = list(self.log_buffer)
            torch.save(data, tmp_path)
            os.replace(tmp_path, self.log_trajectories_file)
            self.log_buffer.clear()
            if self.debug_prints:
                print(f"[LOG] Flushed to {self.log_trajectories_file} at step {self.control_steps}")
        
    ################################################################################################################################

    def reset_idx(self, ids: torch.Tensor) -> None:
        ################# SIM #################
        self.root_states[ids] = torch.zeros((len(ids), 13), dtype=torch.float32, device=self.device)
        self.root_states[ids, 3:7] = sample_random_quaternion_batch(self.device, len(ids))
        idx32 = ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.actor_root_state, gymtorch.unwrap_tensor(idx32), len(idx32)
        )
        #######################################

        self.prev_angvel[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)
        self.delta_actions[ids] = torch.zeros((len(ids), 3), dtype=torch.float, device=self.device)

        self.goal_quat[ids] = sample_random_quaternion_batch(self.device, len(ids))

        self.progress_buf[ids] = 0
        self.reset_buf[ids] = False
        self.timeout_buf[ids] = False

        self.rew_buf[ids] = 0.0

        ###################################################
        if self.randomize:
            self.apply_randomizations(ids, self.dr_params)
        ###################################################

    ################################################################################################################################
                
    def termination(self) -> None:
        self.reset_ids  = torch.nonzero(self.reset_buf, as_tuple=False).flatten()
        if len(self.reset_ids) > 0:
            self.reset_idx(self.reset_ids)
    
    def apply_torque(self) -> None:
        self.actions = torch.mul(self.actions, self.torque_scale)

        self.actions[self.reset_ids] = torch.zeros((len(self.reset_ids), 3), dtype=torch.float, device=self.device)
        
        #########################################
        assert not torch.isnan(self.actions).any(), f"actions has NaN: {self.actions, self.states_buf}"
        assert not torch.isinf(self.actions).any(), f"actions has Inf: {self.actions, self.states_buf}"
        #########################################

        #########################################
        if self.is_eval and self.explosion and self.control_steps == self.explosion_time:
            self.impulse = self.explosion_impulse()
            self.torque_tensor[self.root_indices] = torch.add(self.actions, self.impulse)
        else:
            self.torque_tensor[self.root_indices] = self.actions
        #########################################

        ################## SIM ##################
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.force_tensor),  
            gymtorch.unwrap_tensor(self.torque_tensor), 
            gymapi.LOCAL_SPACE,
        )
        #########################################

        #########################################
        if self.is_eval and self.explosion and self.control_steps == self.explosion_time:
            self.impulse = 0.0
        #########################################
                
    def compute_observations(self) -> None:
        ################# SIM #################
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.satellite_angacc = torch.div(
            torch.sub(self.satellite_angvels, self.prev_angvel),
            self.dt
        )

        self.prev_angvel = self.satellite_angvels.detach().clone()
        self.obs_buf = torch.cat(
            (self.satellite_quats, quat_diff(self.satellite_quats, self.goal_quat), quat_diff_rad(self.satellite_quats, self.goal_quat).unsqueeze(-1), 
                self.satellite_angacc, self.actions), dim=-1)
        self.states_buf = torch.cat(
            (self.obs_buf, self.satellite_angvels), dim=-1)
        ########################################

        ########################################
        assert not torch.isnan(self.obs_buf).any(), f"self.obs_buf has NaN: {self.actions, self.obs_buf}"
        assert not torch.isinf(self.obs_buf).any(), f"self.obs_buf has Inf: {self.actions, self.obs_buf}"
        assert not torch.isnan(self.states_buf).any(), f"self.states_buf has NaN: {self.actions, self.states_buf}"
        assert not torch.isinf(self.states_buf).any(), f"self.states_buf has Inf: {self.actions, self.states_buf}"
        ########################################

    def compute_reward(self) -> None:
        self.rew_buf = self.reward_fn.compute(
            self.satellite_quats, self.satellite_angvels, self.satellite_angacc,
            self.goal_quat, self.actions
        )

    def check_termination(self) -> None:
        timeout = torch.ge(self.progress_buf, self.max_episode_length)

        self.timeout_buf = timeout
        self.reset_buf = timeout

    def pre_physics_step(self, actions):
        if hasattr(self, 'actions'): self.delta_actions = actions.to(self.device) - self.actions
        self.actions = actions.to(self.device)

        self.termination()

        self.apply_torque()

    def post_physics_step(self):
        self.progress_buf = torch.add(self.progress_buf, 1)
        
        self.compute_observations()

        self.compute_reward()

        self.check_termination()

        if self.log_status:
            self.log_status_func()
        
        if self.is_eval and self.log_trajectories:
            self.log_trajectories_func()

        if self.debug_arrows:
            self.draw_arrows()


