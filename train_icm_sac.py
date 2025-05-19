# train_icm_sac.py
from stable_baselines3 import SAC
from fortress_env import FortressEnv
from fortress_env_icm import ICMWrapper
from icm_module import ICM

import torch
import os

# 💻 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_timesteps = 100_000  # 학습 시간 (프레임 수 기준)
save_path = "./models/sac_icm_model"

# 1. 🏗️ 환경 생성 (마우스 조작 + 이미지 상태 제공)
base_env = FortressEnv()

# 2. 🧠 ICM 모델 생성 (상태 전이 예측 기반 보상 생성기)
icm = ICM(obs_shape=(3, 96, 96), action_dim=4).to(device)

# 3. 🔁 보상 래퍼 환경으로 감싸기
env = ICMWrapper(base_env, icm, device=device)

# 4. 🤖 SAC 모델 생성 (CNN 기반 정책 사용)
model = SAC("CnnPolicy", env, verbose=1, device=device)

# 5. 📈 학습 실행
model.learn(total_timesteps=total_timesteps)

# 6. 💾 모델 저장
os.makedirs(save_path, exist_ok=True)
model.save(f"{save_path}/sac_icm")
print("✅ 학습 완료 및 모델 저장 완료!")
