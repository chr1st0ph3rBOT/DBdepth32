# 🔧 필수 라이브러리
import cv2
import numpy as np
from PIL import Image
import pyautogui
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 🎯 CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 중인 디바이스:", device)

# 🎯 마우스 클릭 영역 (너의 좌표 기준)
X_MIN, X_MAX = 169, 418
Y_FIXED = 168
MAX_CLICK_DURATION = 1.0

# 🎯 캡처 함수 (로컬에서는 실제 게임 캡처 코드 연결)
def get_latest_capture():
    return Image.new('RGB', (800, 600), color='white')  # 테스트용

# 🎯 상태 추출 (HP, 바람)
def extract_state(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # AI HP 추출 (Player)
    ai_hp_bar = img[33:54, 35:276]
    ai_hp_ratio = np.mean(ai_hp_bar[:, :, 0]) / 255.0 * 100  # 0~100%

    # Enemy HP 추출
    enemy_hp_bar = img[33:54, 277:514]
    enemy_hp_ratio = np.mean(enemy_hp_bar[:, :, 0]) / 255.0 * 100

    # 바람 정보 추출 (윈드 게이지)
    wind_area = hsv[80:96, 226:324]
    mean_hue = np.mean(wind_area[:, :, 0])
    wind_direction = 1 if mean_hue > 90 else -1  # Hue 기준 (테스트 필요)
    wind_strength = np.std(wind_area[:, :, 2]) / 255.0 * 10

    return np.array([ai_hp_ratio, enemy_hp_ratio, wind_direction, wind_strength], dtype=np.float32)


# 🎯 게임 종료 감지 (화이트 플래시)
def is_game_done(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return np.mean(gray) > 240

# 🎯 보상 함수 (HP 변화 + 실패 클릭 보상)
def compute_reward(prev_state, curr_state, is_done, clicked):
    reward = (prev_state[1] - curr_state[1]) * 10 + (curr_state[0] - prev_state[0]) * -5
    if is_done:
        reward += 1000 if curr_state[1] <= 0 else -1000
    if (prev_state[1] - curr_state[1]) >= 20:
        reward += 50
    if clicked and (prev_state[1] - curr_state[1]) == 0:
        reward -= 10  # 클릭 실패 페널티
    return reward

# 🎯 마우스 실행 (X 위치 + 고정 Y, 클릭 시간)
def execute_action(action_np):
    pyautogui.moveTo(int(action_np[0]), int(action_np[1]))
    pyautogui.mouseDown()
    time.sleep(action_np[2])
    pyautogui.mouseUp()

# 🎯 PPO 모델
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, state):
        return self.net(state)

# 🎯 PPO 설정
STATE_DIM, ACTION_DIM = 4, 1  # 턴 제거 → 4개 상태 입력
LR, GAMMA, EPS_CLIP, UPDATE_EPOCHS = 3e-4, 0.99, 0.2, 5
actor, critic = Actor(STATE_DIM, ACTION_DIM).to(device), Critic(STATE_DIM).to(device)
actor_optim, critic_optim = optim.Adam(actor.parameters(), lr=LR), optim.Adam(critic.parameters(), lr=LR)

# 🎯 데이터 버퍼
class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards, self.log_probs, self.dones = [], [], [], [], []
    def clear(self):
        self.__init__()
buffer = RolloutBuffer()

# 🎯 Advantage 계산
def compute_advantages(rewards, dones, values, gamma=GAMMA):
    advantages, returns, gae, next_value = [], [], 0, 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * 0.95 * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
        returns.insert(0, gae + values[step])
    return torch.FloatTensor(advantages).to(device), torch.FloatTensor(returns).to(device)

# 🎯 에피소드별 리포트 저장
episode_rewards = []

# 🔁 메인 루프
for episode in range(1000):
    buffer.clear()
    state = extract_state(get_latest_capture())
    total_reward = 0

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean_action = actor(state_tensor)
        std = 0.1
        dist = torch.distributions.Normal(mean_action, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # 🎯 X 위치 변환 + Y 고정 + 클릭 시간
        action_np = action.squeeze(0).detach().cpu().numpy()
        x_pos = (action_np[0] + 1) / 2 * (X_MAX - X_MIN) + X_MIN
        y_pos = Y_FIXED
        click_duration = (action_np[0] + 1) / 2 * MAX_CLICK_DURATION
        click_duration = max(0.05, click_duration)

        execute_action([x_pos, y_pos, click_duration])

        next_img = get_latest_capture()
        next_state = extract_state(next_img)
        done = is_game_done(next_img)
        reward = compute_reward(state, next_state, done, clicked=True)
        total_reward += reward

        buffer.states.append(state)
        buffer.actions.append(action.detach().cpu().numpy())
        buffer.rewards.append(reward)
        buffer.log_probs.append(log_prob.detach().cpu().numpy())
        buffer.dones.append(done)

        state = next_state

        if done:
            print(f"[Episode {episode}] Total Reward: {total_reward:.2f}")
            episode_rewards.append(total_reward)
            break

    # PPO 업데이트
    states = torch.FloatTensor(buffer.states).to(device)
    actions = torch.FloatTensor(buffer.actions).to(device)
    old_log_probs = torch.FloatTensor(buffer.log_probs).unsqueeze(-1).to(device)
    rewards, dones = buffer.rewards, buffer.dones
    values = critic(states).detach().squeeze(-1).cpu().numpy()
    advantages, returns = compute_advantages(rewards, dones, values)

    for _ in range(UPDATE_EPOCHS):
        mean_action = actor(states)
        dist = torch.distributions.Normal(mean_action, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages.unsqueeze(-1)
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(critic(states).squeeze(-1), returns)

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

    # 🎯 에피소드별 리포트 그래프
    if episode % 10 == 0 and episode > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(episode_rewards, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("PPO Training Progress")
        plt.legend()
        plt.show()
