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
import pygetwindow as gw
import win32gui
import os
from torch.utils.tensorboard import SummaryWriter

# ✅ CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 중인 디바이스:", device)

# ✅ TensorBoard
writer = SummaryWriter("runs/ppo_experiment")

# ✅ 로그 파일
log_file = open("ppo_log.txt", "a")

# ✅ 클라이언트 좌표 계산
def get_client_area():
    windows = gw.getWindowsWithTitle("Flash Player")
    if not windows:
        raise Exception("Flash Player 창을 찾을 수 없습니다.")
    win = windows[0]
    hwnd = win._hWnd
    left_top = win32gui.ClientToScreen(hwnd, (0, 0))
    right_bottom = win32gui.ClientToScreen(hwnd, win32gui.GetClientRect(hwnd)[2:])
    left, top = left_top
    right, bottom = right_bottom
    width = right - left
    height = bottom - top
    return left, top, width, height

# ✅ 캡처 이미지 불러오기
SAVE_DIR = "window_captures"
def get_latest_capture():
    png_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".png")]
    if not png_files:
        return None
    latest_file = max(png_files, key=lambda x: os.path.getmtime(os.path.join(SAVE_DIR, x)))
    return Image.open(os.path.join(SAVE_DIR, latest_file))

# ✅ 내 턴 확인 (임시)
def is_my_turn():
    return random.choice([True, False])

# ✅ 바람 정보 추출
def extract_wind_info(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    bar_area = img[96:112, 226:324]
    mask_yellow = cv2.inRange(bar_area, (20, 100, 100), (40, 255, 255))
    coords = cv2.findNonZero(mask_yellow)
    if coords is not None:
        mean_x = np.mean(coords[:, 0, 0])
        center_x = mask_yellow.shape[1] / 2
        wind_direction = -1 if mean_x < center_x else 1
        distance_ratio = abs(mean_x - center_x) / center_x
        wind_strength = distance_ratio * 10
    else:
        wind_direction, wind_strength = 0, 0
    print(f"[DEBUG] Wind Dir: {wind_direction}, Strength: {wind_strength:.2f}")
    return wind_direction, wind_strength

# ✅ 상태 추출
def extract_state(image):
    img = np.array(image)
    player_hp = np.mean(img[33:54, 35:276, 0]) / 255.0 * 100
    enemy_hp = np.mean(img[33:54, 277:514, 0]) / 255.0 * 100
    wind_direction, wind_strength = extract_wind_info(image)
    return np.array([player_hp, enemy_hp, wind_direction, wind_strength], dtype=np.float32)

# ✅ 게임 종료 감지
def is_game_done(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return np.mean(gray) > 240

# ✅ 보상 함수
def compute_reward(prev_state, curr_state, is_done, clicked):
    reward = (prev_state[1] - curr_state[1]) * 10 + (curr_state[0] - prev_state[0]) * -5
    if is_done:
        reward += 1000 if curr_state[1] <= 0 else -1000
    if (prev_state[1] - curr_state[1]) >= 20:
        reward += 50
    if clicked and (prev_state[1] - curr_state[1]) == 0:
        reward -= 10
    return reward

# ✅ 마우스 좌표 변환
def map_action_to_client_coords(action_value, client_left, client_top, client_width, client_height):
    x_ratio = (action_value[0] + 1) / 2
    y_ratio = 0.5
    x_pos = int(x_ratio * client_width)
    y_pos = int(y_ratio * client_height)
    return client_left + x_pos, client_top + y_pos

# ✅ 마우스 실행
def execute_action(action_np, client_left, client_top, client_width, client_height):
    final_x, final_y = map_action_to_client_coords(action_np, client_left, client_top, client_width, client_height)
    pyautogui.moveTo(final_x, final_y)
    pyautogui.mouseDown()
    time.sleep(action_np[1])
    pyautogui.mouseUp()
    print(f"[DEBUG] Mouse: ({final_x}, {final_y}) | Click Time: {action_np[1]:.2f}")

# ✅ PPO 모델
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

STATE_DIM, ACTION_DIM = 4, 1
LR, GAMMA, EPS_CLIP, UPDATE_EPOCHS = 3e-4, 0.99, 0.2, 5
actor, critic = Actor(STATE_DIM, ACTION_DIM).to(device), Critic(STATE_DIM).to(device)
actor_optim, critic_optim = optim.Adam(actor.parameters(), lr=LR), optim.Adam(critic.parameters(), lr=LR)

class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards, self.log_probs, self.dones = [], [], [], [], []
    def clear(self):
        self.__init__()

buffer = RolloutBuffer()

def compute_advantages(rewards, dones, values, gamma=GAMMA):
    advantages, returns, gae, next_value = [], [], 0, 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * 0.95 * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
        returns.insert(0, gae + values[step])
    return torch.FloatTensor(advantages).to(device), torch.FloatTensor(returns).to(device)

episode_rewards = []

# ✅ 메인 루프
for episode in range(1000):
    buffer.clear()
    state = extract_state(get_latest_capture())
    total_reward = 0

    client_left, client_top, client_width, client_height = get_client_area()

    while True:
        done = is_game_done(get_latest_capture())
        if done:
            print(f"[EP {episode}] Total Reward: {total_reward:.2f}")
            log_file.write(f"EP {episode}, Total Reward: {total_reward:.2f}\n")
            log_file.flush()
            writer.add_scalar("Total Reward", total_reward, episode)
            episode_rewards.append(total_reward)
            break

        if is_my_turn():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            mean_action = actor(state_tensor)
            std = 0.1
            dist = torch.distributions.Normal(mean_action, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            action_np = action.squeeze(0).detach().cpu().numpy()
            click_duration = (action_np[0] + 1) / 2 * 1.0
            click_duration = max(0.05, click_duration)

            execute_action([action_np[0], click_duration], client_left, client_top, client_width, client_height)

            next_img = get_latest_capture()
            next_state = extract_state(next_img)
            reward = compute_reward(state, next_state, done, clicked=True)
            total_reward += reward

            print(f"[DEBUG] Reward: {reward:.2f}, Total: {total_reward:.2f}")

            buffer.states.append(state)
            buffer.actions.append(action.detach().cpu().numpy())
            buffer.rewards.append(reward)
            buffer.log_probs.append(log_prob.detach().cpu().numpy())
            buffer.dones.append(done)

            state = next_state
        else:
            state = extract_state(get_latest_capture())

    if buffer.states:
        states = torch.FloatTensor(buffer.states).to(device)
        actions = torch.FloatTensor(buffer.actions).to(device)
        old_log_probs = torch.FloatTensor(buffer.log_probs).unsqueeze(-1).to(device)
        values = critic(states).detach().squeeze(-1).cpu().numpy()
        advantages, returns = compute_advantages(buffer.rewards, buffer.dones, values)

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

        writer.add_scalar("Actor Loss", actor_loss.item(), episode)
        writer.add_scalar("Critic Loss", critic_loss.item(), episode)

    if episode % 10 == 0 and episode > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(episode_rewards, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("PPO Training Progress")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

log_file.close()
writer.close()
