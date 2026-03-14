import numpy as np
import gymnasium as gym
import torch
import DES_ambo as DES_ambo
from stable_baselines3 import PPO
import pandas as pd
import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv
from multiprocessing import cpu_count
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import pickle
import torch.multiprocessing as mp

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

def linear_schedule(initial_learing_rate = 5e-4, final_learing_rate = 5e-6):
    """
    decay linerlly and holds constant at a certain value
    """
    def func(progress_remaining):
        return max(progress_remaining * initial_learing_rate,final_learing_rate)
    return func


def make_env(env_id: str, rank: int, seed: int = 0, env_kwargs: dict = {}):
    """
    Factory function for multiprocessing
    """
    def _init():
        env = gym.make(env_id, **env_kwargs)
        # use a different seed for each env
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


if __name__ == "__main__":
    
    
    # Use 'spawn' instead of the default 'fork' to start subprocesses.
    # Each subprocess starts as a fresh Python interpreter and builds its own env cleanly.
    # Required to avoid CUDA/PyTorch crashes that occur when forking a process with GPU state.
    mp.set_start_method('spawn', force=True)
    
    
    #Data Loading 
    file_paths = {
        "accident_rate": "final_version/data/accident_rate.csv",
        "distance_Base_to_Incident_df": "final_version/data/distance_base_to_incident.csv",
        "distance_Hospital_to_Base_df": "final_version/data/distance_hospital_to_base.csv",
        "nearest_place" : "final_version/data/nearest_places_data.csv",
        "ambulance_initialization" : "final_version/data/ambulance_initialization.csv"
    }
    # Load data
    print("Loading data...")
    accident_rate = pd.read_csv(file_paths["accident_rate"])
    accident_rate = accident_rate /2
    distance_Base_to_Incident_df = pd.read_csv(file_paths["distance_Base_to_Incident_df"])
    distance_Hospital_to_Base_df = pd.read_csv(file_paths["distance_Hospital_to_Base_df"])
    nearest_place = pd.read_csv(file_paths["nearest_place"])
    ambulance_initialization = pd.read_csv(file_paths["ambulance_initialization"])
    ambulance_initialization_dict = ambulance_initialization.to_dict()['initial_ambulances']
    with open('incident_pred.pkl', 'rb') as f:
        accident_rate_pred = pickle.load(f)
    print(np.array(accident_rate_pred).shape)
    print("Data loading complete.")

    # Env Set up
    print("Creating environment...")
    env_id = "DES_ambo/DES_ambo_map-train"
    env_kwargs = dict(
        accident_rate = accident_rate,
        accident_rate_pred = accident_rate_pred,
        distance_Base_to_Incident_df = distance_Base_to_Incident_df,
        distance_Hospital_to_Base_df = distance_Hospital_to_Base_df,
        nearest_place=nearest_place,
        init_ambulances_per_base_dict=ambulance_initialization_dict,
        run_until=1440,
        trace=False,
        test = False
    )
    
    num_cpu = cpu_count()
    print(f"Using {num_cpu} parallel CPU processes for environment simulation.")
    
    # Create the SubprocVecEnv 
    env = SubprocVecEnv([make_env(env_id, i, env_kwargs=env_kwargs) for i in range(num_cpu)])
    
    MODEL_NAME = "35M_run"
    MODEL_SAVE_PATH = f"/home/thurein/ambo_allocate/integrate_map/{MODEL_NAME}"
    STATS_SAVE_PATH = f"/home/thurein/ambo_allocate/integrate_map/{MODEL_NAME}_stat.pkl"

    norm_env = VecNormalize(env, norm_obs=False, norm_reward=True)

    model = PPO(
        "MultiInputPolicy",
        norm_env,
        learning_rate=linear_schedule(),
        n_steps=8192,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,
        vf_coef=1.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512])),
        verbose=1,
        device=device,
        tensorboard_log="./ppo_des_ambo_tensorboard/"
    )

    print("Starting training for 35M steps...")
    model.learn(
        total_timesteps=35_000_000,
        tb_log_name=MODEL_NAME,
    )

    print("Training complete.")
    model.save(MODEL_SAVE_PATH)
    norm_env.save(STATS_SAVE_PATH)