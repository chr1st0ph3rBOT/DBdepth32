import cv2
import numpy as np
from PIL import Image
import pyautogui
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 중인 디바이스:", device)

# 캡처 모듈 (Colab에서는 실제 게임 화면 캡처 불가, 로컬에서만 실행)
def get_latest_capture():
    # 로컬에서는 게임 캡처 코드 연결 필요
    return Image.new('RGB', (800, 600), color='white')  # 테스트용

def extract_state(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    player_hp_bar = img[550:560, 100:300]
    enemy_hp_bar = img[550:560, 500:700]
    player_hp_ratio = np.mean(player_hp_bar[:, :, 0]) / 255.0
    enemy_hp_ratio = np.mean(enemy_hp_bar[:, :, 0]) / 255.0
    wind_area = hsv[50:80, 350:450]
    mean_hue = np.mean(wind_area[:, :, 0])
    wind_direction = 1 if mean_hue > 90 else -1
    wind_strength = np.std(wind_area[:, :, 2]) / 255.0 * 10
    chance_button = img[500:520, 50:70]
    chance_color = np.mean(chance_button)
    is_my_turn = 1 if chance_color > 100 else 0
    return np.array([player_hp_ratio * 100, enemy_hp_ratio * 100, wind_direction, wind_strength, is_my_turn], dtype=np.float32)

def is_game_done(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > 240

def compute_reward(prev_state, curr_state, is_done):
    reward = 0
    enemy_hp_diff = prev_state[1] - curr_state[1]
    player_hp_diff = curr_state[0] - prev_state[0]
    reward += enemy_hp_diff * 10
    reward += player_hp_diff * -5
    if is_done:
        if curr_state[1] <= 0:
            reward += 1000
        elif curr_state[0] <= 0:
            reward -= 1000
    if enemy_hp_diff >= 20:
        reward += 50
    return reward

def execute_action(action):
    x, y, click_duration = action
    pyautogui.moveTo(int(x), int(y))
    pyautogui.mouseDown()
    time.sleep(click_duration)
    pyautogui.mouseUp()

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

STATE_DIM = 5
ACTION_DIM = 3
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
UPDATE_EPOCHS = 5

actor = Actor(STATE_DIM, ACTION_DIM).to(device)
critic = Critic(STATE_DIM).to(device)
actor_optim = optim.Adam(actor.parameters(), lr=LR)
critic_optim = optim.Adam(critic.parameters(), lr=LR)

class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards, self.log_probs, self.dones = [], [], [], [], []
    def clear(self):
        self.__init__()
buffer = RolloutBuffer()

def compute_advantages(rewards, dones, values, gamma=GAMMA):
    advantages, returns = [], []
    gae, next_value = 0, 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * 0.95 * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
        returns.insert(0, gae + values[step])
    return torch.FloatTensor(advantages).to(device), torch.FloatTensor(returns).to(device)

for episode in range(10000):
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

        action_np = action.squeeze(0).detach().cpu().numpy()
        action_np[0] = (action_np[0] + 1) / 2 * 800
        action_np[1] = (action_np[1] + 1) / 2 * 600
        action_np[2] = max(0.05, action_np[2] * 1.0)

        execute_action(action_np)

        next_img = get_latest_capture()
        next_state = extract_state(next_img)
        done = is_game_done(next_img)
        reward = compute_reward(state, next_state, done)
        total_reward += reward

        buffer.states.append(state)
        buffer.actions.append(action.squeeze(0).detach().cpu().numpy())
        buffer.rewards.append(reward)
        buffer.log_probs.append(log_prob.detach().cpu().numpy())
        buffer.dones.append(done)

        state = next_state

        if done:
            print(f"[Episode {episode}] Total Reward: {total_reward:.2f}")
            break

    states = torch.FloatTensor(buffer.states).to(device)
    actions = torch.FloatTensor(buffer.actions).to(device)
    old_log_probs = torch.FloatTensor(buffer.log_probs).unsqueeze(-1).to(device)
    rewards = buffer.rewards
    dones = buffer.dones
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
