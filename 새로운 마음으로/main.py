import time
import numpy as np
from modules.image_capture import get_latest_capture, start_capture_loop
from modules.preprocessing import preprocess
from modules.cnn_module import CNNModel
from modules.rl_agent import SACAgent, ReplayBuffer
from modules.reward_module import RewardModule
from modules.action_executor import ActionExecutor

# 설정
MAX_EPISODES = 100000
EXPLORATION_STEPS = 1000
BATCH_SIZE = 64
TRAIN_EVERY = 10
REPLAY_SIZE = 10000

# 초기화
cnn = CNNModel()
agent = SACAgent()
buffer = ReplayBuffer(REPLAY_SIZE, state_dim=256, action_dim=3)
executor = ActionExecutor()
rewarder = RewardModule()

# 캡처 루프 시작
start_capture_loop()
time.sleep(1.0)

prev_hp = 100
prev_enemy_hp = 100

episode = 0
steps = 0

while episode < MAX_EPISODES:
    img = get_latest_capture()
    if img is None:
        continue

    # 상태 벡터
    state = preprocess(img)
    state = np.expand_dims(state, axis=0)
    state_vector = cnn(state).numpy()[0]

    # 행동 선택
    if steps < EXPLORATION_STEPS:
        action = np.random.uniform(-1, 1, size=3)
    else:
        action = agent.select_action(state_vector)

    # 행동 수행
    executor.execute(action)
    time.sleep(0.1)

    # 다음 상태
    next_img = get_latest_capture()
    if next_img is None:
        continue

    next_state = preprocess(next_img)
    next_state = np.expand_dims(next_state, axis=0)
    next_state_vector = cnn(next_state).numpy()[0]

    # 보상 계산 (가상)
    my_hp_change = np.random.randint(-5, 1)
    enemy_hp_change = np.random.randint(-30, -5)
    item_count = np.random.randint(0, 3)
    time_bonus = np.random.uniform(0.1, 1.0)

    reward = rewarder.external_reward_func(
        my_hp_change=my_hp_change,
        enemy_hp_change=enemy_hp_change,
        item_count=item_count,
        time_seconds=time_bonus
    )

    done = False
    buffer.add(state_vector, action, reward, next_state_vector, done)

    if steps > EXPLORATION_STEPS and steps % TRAIN_EVERY == 0:
        batch = buffer.sample(BATCH_SIZE)
        agent.update(batch)

    steps += 1
    episode += 1

    print(f"[Ep {episode}] Reward: {reward:.2f}, HP Δ:({my_hp_change}/{enemy_hp_change}), Item: {item_count}")

print("학습 종료")
