# vec_task.py

import isaacgym #BugFix
import torch
from isaacgym import gymtorch, gymapi

import os
import time
import sys
import random
import numpy as np
from datetime import datetime
from os.path import join
from typing import Dict, Any, Tuple
from abc import ABC

from gym import spaces

from torch.profiler import record_function

from code.utils.satellite_util import quat_mul, quat_conjugate
import math

EXISTING_SIM = None

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        print("Using EXISTING Sim Instance")
        return EXISTING_SIM
    else:
        print("Creating NEW Sim Instance")
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class Env(ABC):
    def __init__(self, config: Dict[str, Any], rl_device: str, sim_device: str, graphics_device_id: int, headless: bool):
        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        self.headless = headless

        enable_camera_sensors = config["env"].get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments
        
        self.num_observations = config["env"].get("numObservations", 0)
        self.num_states = config["env"].get("numStates", 0)

        self.observation_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf, dtype=np.float64)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf, dtype=np.float64)

        self.num_actions = config["env"]["numActions"]
        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1., dtype=np.float64)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

        self.control_steps: int = 0

        self.render_fps: int = config["env"].get("renderFPS", -1)
        self.last_frame_time: float = 0.0

        self.record_frames: bool = False
        self.record_frames_dir = join("recorded_frames", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

class VecTask(Env):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless): 
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)
        self.sim_params = self.__parse_sim_params(config["physics_engine"], config["sim"])
        if config["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif config["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {config['physics_engine']}"
            raise ValueError(msg)

        self.dt: float = self.sim_params.dt

        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True
        
        self.set_viewer()
        self.allocate_buffers()

        self.obs_states_dict = {}

    def set_viewer(self):
        self.enable_viewer_sync = True
        self.viewer = None

        if self.headless == False:
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames")

            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def allocate_buffers(self):
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        with record_function("#VecTask__STEP"):

            ##########################################################################################
            if self.randomize and self.dr_params.get("actions", {}).get("enabled", False):
                if self.debug_prints: print(f"Action BEFORE randomization -> actions[0]: {', '.join(f'{v:.2f}' for v in actions[0].tolist())}")
                actions = self.dr_randomizations['actions']['noise_lambda'](actions)
                if self.debug_prints: print(f"Action AFTER randomization -> actions[0]: {', '.join(f'{v:.2f}' for v in actions[0].tolist())}")
            ##########################################################################################

            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
            
            if self.debug_prints:
                print("#" * 50)
                print(f"Actions         MAX: {actions.max().item():.2f} MIN: {actions.min().item():.2f} MEAN: {actions.mean().item():.2f} STD: {actions.std().item():.2f}")  # Debugging output

            with record_function("$VecTask__step__pre_physics_step"):
                self.pre_physics_step(actions)

            for i in range(self.control_freq_inv):
                with record_function("#VecTask__step__SIM"):
                    self.gym.simulate(self.sim)

            if self.device == 'cpu':
                with record_function("$VecTask__step__FETCH_RESULTS"):
                    self.gym.fetch_results(self.sim, True)

            with record_function("$VecTask__step__post_physics_step"):
                self.post_physics_step()

            ##########################################################################################
            if self.randomize and self.dr_params.get("observations", {}).get("enabled", False):
                if self.debug_prints: print(f"Observations BEFORE randomization -> obs_buf[0]: {', '.join(f'{v:.2f}' for v in self.obs_buf[0].tolist())}")
                self.obs_buf = self.apply_noise_on_custom_buffer(self.obs_buf, 'observations')
                if self.debug_prints: print(f"Observations AFTER randomization -> obs_buf[0]: {', '.join(f'{v:.2f}' for v in self.obs_buf[0].tolist())}")
            ##########################################################################################

            ##########################################################################################
            if self.randomize and self.dr_params.get("states", {}).get("enabled", False):
                if self.debug_prints: print(f"States BEFORE randomization -> state_buf[0]: {', '.join(f'{v:.2f}' for v in self.states_buf[0].tolist())}")
                self.states_buf = self.apply_noise_on_custom_buffer(self.states_buf, 'states')
                if self.debug_prints: print(f"States AFTER randomization -> state_buf[0]: {', '.join(f'{v:.2f}' for v in self.states_buf[0].tolist())}")
            ##########################################################################################

            self.obs_states_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            self.obs_states_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            
            self.control_steps += 1
            self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

            if self.debug_prints:
                print("#" * 50)
                num_quats = 4; num_quat_diff = 4; num_quat_diff_rad = 1; num_angacc = 3; num_actions = 3; num_angvels = 3
                l_index = 0; h_index = num_quats
                print(f"Quats           MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                l_index = num_quats; h_index = num_quats + num_quat_diff
                print(f"QuatsDiff       MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                l_index = num_quats + num_quat_diff; h_index = num_quats + num_quat_diff + num_quat_diff_rad
                print(f"QuatsDiffRad    MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                l_index = num_quats + num_quat_diff + num_quat_diff_rad; h_index = num_quats + num_quat_diff + num_quat_diff_rad + num_angacc
                print(f"AngAcc          MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                l_index = num_quats + num_quat_diff + num_quat_diff_rad + num_angacc; h_index = num_quats + num_quat_diff + num_quat_diff_rad + num_angacc + num_actions
                print(f"Act             MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                l_index = num_quats + num_quat_diff + num_quat_diff_rad + num_angacc + num_actions; h_index = num_quats + num_quat_diff + num_quat_diff_rad + num_angacc + num_actions + num_angvels
                print(f"AngVels         MAX: {self.obs_states_dict['states'][:, l_index:h_index].max().item():.2f} MIN: {self.obs_states_dict['states'][:, l_index:h_index].min().item():.2f} MEAN: {self.obs_states_dict['states'][:, l_index:h_index].mean().item():.2f} STD: {self.obs_states_dict['states'][:, l_index:h_index].std().item():.2f}")  # Debugging output
                print(f"Reward          MAX: {self.rew_buf.max().item():.2f} MIN: {self.rew_buf.min().item():.2f} MEAN: {self.rew_buf.mean().item():.2f} STD: {self.rew_buf.std().item():.2f}")  # Debugging output

                print(f"Timeouts:       {self.timeout_buf.sum().item()}")  # Debugging output
                print(f"Reset:          {self.reset_buf.sum().item()}")  # Debugging output
                print(f"Extras:         {self.extras}")  # Debugging output
                print(f"Steps:          {self.control_steps}")  # Debugging output

        return self.obs_states_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def reset(self):
        self.obs_states_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        self.obs_states_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return self.obs_states_dict

    def render(self):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                self.gym.sync_frame_time(self.sim)

                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    render_dt = self.dt * self.control_freq_inv
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"frame_{self.control_steps}.png"))

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        sim_params = gymapi.SimParams()

        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        if physics_engine == "physx":
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        return sim_params

class DRVecTask(VecTask):
    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless):
        ###################################################
        self.randomize = config["dr_randomization"].get("enabled", False)
        self.dr_params = config["dr_randomization"].get("dr_params", {})
        ###################################################
        if self.randomize: 
            self.first_randomization = True
            self.original_props = {}
            ##################################################
            self.dr_randomizations = {}

        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)

        self.q_null = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

    #############################################################################################
    
    def apply_noise_on_custom_buffer(self, buf, buf_type):
        q_noise = self.dr_randomizations[buf_type]['noise_lambda_quat'](self.q_null.clone())
        buf[:, 0:4] = quat_mul(q_noise, buf[:, 0:4])
        buf[:, 4:8] = quat_mul(q_noise, buf[:, 4:8])
        buf[:, 8] = buf[:, 8] + 2.0 * torch.asin(q_noise[:, 0:3].norm(dim=-1).clamp(max=1.0))
        buf[:, 9:] = self.dr_randomizations[buf_type]['noise_lambda'](buf[:, 9:])

        return buf

    def _init_randomization_functions(self, dr_params):
        for param in ["observations", "states", "actions"]:

            if not dr_params.get(param, {}).get("enabled"):
                self.dr_randomizations[param] = {"noise_lambda": lambda x: x, "noise_lambda_quat": lambda q: q}
                continue
            
            dist = dr_params[param]["distribution"]
            operation = dr_params[param]["operation"]

            if dist == 'gaussian':
                mu, std = dr_params[param]["range"]
                q_noise_gaussian = dr_params[param].get("quaternion_noise", None)

                if operation == "scaling":
                    def noise_lambda(tensor, mu=mu, std=std):
                        return tensor * (torch.randn_like(tensor) * std + mu)
                elif operation == "addition":
                    def noise_lambda(tensor, mu=mu, std=std):
                        return tensor + (torch.randn_like(tensor) * std + mu)
                else:
                    raise ValueError("Unsupported operation type")
                
                def noise_lambda_quat(q, q_noise_gaussian=q_noise_gaussian):
                    axis = torch.randn((q.shape[0], 3), device=q.device, dtype=q.dtype)
                    axis = axis / axis.norm(dim=-1, keepdim=True)
                    angle = torch.randn((q.shape[0], 1), device=q.device, dtype=q.dtype) * (q_noise_gaussian * math.pi)
                    dq = torch.cat([axis * torch.sin(0.5 * angle), torch.cos(0.5 * angle)], dim=-1)
                    return quat_mul(dq, q)
            
            elif dist == 'uniform':
                lo, hi = dr_params[param]["range"]
                q_noise_uniform = dr_params[param].get("quaternion_noise", None)

                if operation == "scaling":
                    def noise_lambda(tensor, lo=lo, hi=hi):
                        return tensor * (torch.rand_like(tensor) * (hi - lo) + lo)
                elif operation == "addition":
                    def noise_lambda(tensor, lo=lo, hi=hi):
                        return tensor + (torch.rand_like(tensor) * (hi - lo) + lo)
                else:
                    raise ValueError("Unsupported operation type")
                
                def noise_lambda_quat(q, q_noise_uniform=q_noise_uniform):
                    axis = torch.randn((q.shape[0], 3), device=q.device, dtype=q.dtype)
                    axis = axis / axis.norm(dim=-1, keepdim=True)
                    angle = torch.rand((q.shape[0], 1), device=q.device, dtype=q.dtype) * (q_noise_uniform * math.pi)
                    dq = torch.cat([axis * torch.sin(0.5 * angle), torch.cos(0.5 * angle)], dim=-1)
                    return quat_mul(dq, q)
                
            else:
                raise ValueError("Unsupported distribution type")
            self.dr_randomizations[param] = {
                "noise_lambda": noise_lambda,
                "noise_lambda_quat": noise_lambda_quat
            }

    #############################################################################################

    def _sample_random_val(self, params):
        if params["distribution"] == "gaussian":
            mu, var = params["range"]
            sample = np.random.normal(mu, var)
        elif params["distribution"] == "uniform":
            lo, hi = params["range"]
            sample = np.random.uniform(lo, hi)
        else:
            raise ValueError(f"Unsupported distribution type")
        return sample

    def _apply_randomization(self, rb_prop, og_attr_val, attr_name, attr_params):
        if isinstance(og_attr_val, gymapi.Mat33):
            rx, ry, rz = og_attr_val.x, og_attr_val.y, og_attr_val.z
            new_val = gymapi.Mat33()
            if attr_params["operation"] == "scaling":
                factor = self._sample_random_val(attr_params)
                new_val.x.x, new_val.x.y, new_val.x.z = rx.x * factor, rx.y * factor, rx.z * factor
                new_val.y.x, new_val.y.y, new_val.y.z = ry.x * factor, ry.y * factor, ry.z * factor
                new_val.z.x, new_val.z.y, new_val.z.z = rz.x * factor, rz.y * factor, rz.z * factor
            elif attr_params["operation"] == "addition":
                delta = self._sample_random_val(attr_params)
                new_val.x.x, new_val.x.y, new_val.x.z = rx.x + delta, rx.y + delta, rx.z + delta
                new_val.y.x, new_val.y.y, new_val.y.z = ry.x + delta, ry.y + delta, ry.z + delta
                new_val.z.x, new_val.z.y, new_val.z.z = rz.x + delta, rz.y + delta, rz.z + delta
            else:
                raise ValueError(f"Unsupported operation type")
        else:
            raise ValueError(f"Unsupported attribute type for randomization")
        setattr(rb_prop, attr_name, new_val)

    def _randomize_actor_properties(self, env_ids, dr_params):
        for env_id in env_ids:
            for actor_name, actor_config  in dr_params["actor_params"].items():
                actor_handle = self.gym.find_actor_handle(self.envs[env_id], actor_name)
                if "color" in actor_config and actor_config ["color"]:
                    num_bodies = self.gym.get_actor_rigid_body_count(self.envs[env_id], actor_handle)
                    for body_index in range(num_bodies):
                        self.gym.set_rigid_body_color(self.envs[env_id], actor_handle, body_index, gymapi.MESH_VISUAL, gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                if "rigid_body_properties" in actor_config:
                    rb_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], actor_handle)
                    ####################################################################
                    if self.first_randomization:
                        self.original_props[actor_name] = [rb.inertia for rb in rb_props]
                    ####################################################################
                    for i, rb_prop in enumerate(rb_props):
                        if "inertia" in actor_config["rigid_body_properties"]:
                            self._apply_randomization(rb_prop, self.original_props[actor_name][i], "inertia", actor_config["rigid_body_properties"]["inertia"])
                    self.gym.set_actor_rigid_body_properties(
                        self.envs[env_id], actor_handle, rb_props, recomputeInertia=True
                    )
                    
    def apply_randomizations(self, env_ids, dr_params):
        if self.first_randomization:
            self._init_randomization_functions(dr_params)

        if dr_params.get("actor_params", {}).get("satellite", {}).get("rigid_body_properties", {}).get("inertia", {}).get("enabled", False):
            self._randomize_actor_properties(env_ids, dr_params)

        self.first_randomization = False

        if self.debug_prints:
            header = f"{'Env':<5} {'Ixx':>10} {'Iyy':>10} {'Izz':>10}"
            print("=" * len(header))
            print(header)
            print("-" * len(header))
            for env_id in env_ids:
                env = self.envs[env_id]
                actor_handle = self.actor_handles[env_id]
                rb_props = self.gym.get_actor_rigid_body_properties(env, actor_handle)
                I = rb_props[0].inertia
                print(f"{env_id:<5} {I.x.x:>10.4f} {I.y.y:>10.4f} {I.z.z:>10.4f}")
            print("=" * len(header) + "\n")