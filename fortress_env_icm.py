# fortress_env_icm.py
# ICM을 이용한 보상 래핑 환경 클래스
import gym
import torch
import numpy as np

class ICMWrapper(gym.Wrapper):
    def __init__(self, env, icm_model, device):
        """
        env: 기존 강화학습 환경 (FortressEnv)
        icm_model: ICM 모듈 (forward_loss 계산용)
        device: 'cuda' or 'cpu'
        """
        super().__init__(env)
        self.icm = icm_model
        self.device = device
        self.last_obs = None  # 이전 상태 저장

    def reset(self):
        """
        환경 초기화 + 초기 상태 저장
        """
        obs = self.env.reset()
        self.last_obs = obs
        return obs

    def step(self, action):
        """
        1. 기존 환경에서 행동 수행
        2. 상태 전이 (s_t → s_{t+1})에 대해 ICM 보상 계산
        """
        obs, _, done, info = self.env.step(action)

        # 상태, 다음 상태, 행동을 Tensor로 변환
        obs_tensor = self._preprocess(self.last_obs)
        next_tensor = self._preprocess(obs)
        action_tensor = torch.tensor([action], dtype=torch.float32).to(self.device)

        # ICM 기반 intrinsic reward 계산 (예측 실패 정도)
        intrinsic_reward = self.icm.forward_loss(obs_tensor, next_tensor, action_tensor).item()

        # 이전 상태 갱신
        self.last_obs = obs

        return obs, intrinsic_reward, done, info

    def _preprocess(self, obs):
        """
        관측 이미지를 Torch Tensor로 변환 + 정규화
        (HWC → CHW, float32, 0~1 스케일)
        """
        obs = np.transpose(obs, (2, 0, 1)) / 255.0
        obs_tensor = torch.tensor([obs], dtype=torch.float32).to(self.device)
        return obs_tensor
