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
from checking_turn import is_game_over as is_my_turn

# ✅ CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 중인 디바이스:", device)

# ✅ 클릭 시간 스케일 (NEW!)
MIN_CLICK = 0.1
MAX_CLICK = 1.1

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

#보상함수
def compute_reward(prev_state, curr_state, is_done, clicked):
    """
    prev_state: 턴 시작 시 상태 [player_hp, enemy_hp, wind_dir, wind_str]
    curr_state: 턴 종료 시 상태
    is_done: 게임 종료 여부
    clicked: 공격을 시도했는지 여부
    """
    # 1) 체력 변화량 계산
    player_hp_diff = prev_state[0] - curr_state[0]
    enemy_hp_diff  = prev_state[1] - curr_state[1]

    # 2) 기본 보상: 적 체력 감소 → +, 내 체력 감소 → –
    reward = enemy_hp_diff * 10 - player_hp_diff * 10

    # 3) 치명타 보너스 (큰 대미지)
    if enemy_hp_diff >= 20:
        reward += 50

    # 4) 공격 시도했지만 대미지 없음 → 페널티
    if clicked and enemy_hp_diff == 0:
        reward -= 10

    # 5) 게임 종료 시 추가 승·패 보상
    if is_done:
        if curr_state[1] <= 0:
            reward += 500   # 승리
        elif curr_state[0] <= 0:
            reward -= 500   # 패배

    return reward


# ✅ 마우스 좌표 변환
def map_action_to_client_coords(action_value,
                                client_left, client_top,
                                client_width, client_height):
    # 1) 윈도우 내부에서 사용할 영역 (픽셀)
    x_min, x_max = 163, 255
    y_min, y_max = 192, 265

    # 2) action_value[0]을 0~1 사이로 정규화
    x_ratio = (action_value[0] + 1) / 2

    # 3) 수평 방향만 비율 매핑
    x_internal = x_min + x_ratio * (x_max - x_min)
    #    수직 방향은 영역 중앙에 고정/--
    y_internal = (y_min + y_max) / 2

    # 4) 스크린 절대 좌표로 변환
    final_x = int(client_left + x_internal)
    final_y = int(client_top  + y_internal)

    return final_x, final_y


# ✅ 마우스 실행
def execute_action(action_np, client_left, client_top, client_width, client_height):
    final_x, final_y = map_action_to_client_coords(action_np, client_left, client_top, client_width, client_height)
    pyautogui.moveTo(final_x, final_y)
    pyautogui.mouseDown()
    safe_click_time = max(0.0, action_np[1])  # 음수 방지
    time.sleep(safe_click_time)
    pyautogui.mouseUp()
    print(f"[DEBUG] Mouse: ({final_x}, {final_y}) | Click Time: {safe_click_time:.2f}")


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
            episode_rewards.append(total_reward)
            break

        if is_my_turn():
            time.sleep(2)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            mean_action = actor(state_tensor)
            std = 0.1
            dist = torch.distributions.Normal(mean_action, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            action_np = action.squeeze(0).detach().cpu().numpy()
            click_duration = (action_np[0] + 1) / 2 * (MAX_CLICK - MIN_CLICK) + MIN_CLICK

            # ⭐ 기존: 클릭 후 바로 보상 계산 (삭제)
            # execute_action 후 상태 저장 & 보상 계산 분리

            # ⭐ 1) 클릭 실행
            execute_action([action_np[0], click_duration], client_left, client_top, client_width, client_height)

            # ⭐ 2) HP 변화 대기 (3초 or 변화 감지)
            def wait_for_hp_change(prev_state, timeout=6.0, check_interval=0.2):
                start_time = time.time()
                while time.time() - start_time < timeout:
                    img = get_latest_capture()
                    curr_state = extract_state(img)

                    if abs(prev_state[1] - curr_state[1]) > 0.5 or abs(prev_state[0] - curr_state[0]) > 0.5:
                        print(f"[HP CHANGE DETECTED] Player HP: {prev_state[0]:.2f} → {curr_state[0]:.2f} | Enemy HP: {prev_state[1]:.2f} → {curr_state[1]:.2f}")
                        return curr_state, True
                    time.sleep(check_interval)
                print("[HP CHANGE DETECTED] No change detected within timeout.")
                return curr_state, False

            # ⭐ 3) 보상 계산
            next_state, hp_changed = wait_for_hp_change(state)
            reward = compute_reward(state, next_state, is_done=False, clicked=True)
            if not hp_changed:
                reward -= 10  # 턴 낭비 패널티

            total_reward += reward
            print(f"[REWARD] {reward:.2f} | Total: {total_reward:.2f}")

            # ⭐ 4) 버퍼 저장
            buffer.states.append(state)
            buffer.actions.append(action.detach().cpu().numpy())
            buffer.rewards.append(reward)
            buffer.log_probs.append(log_prob.detach().cpu().numpy())
            buffer.dones.append(False)

            # ⭐ 5) 상태 업데이트
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
