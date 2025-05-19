import torch

def calculate_reward(state, next_state, action):
    """
    순수한 보상 함수를 정의합니다.
    - state: 현재 상태 (Tensor)
    - next_state: 다음 상태 (Tensor)
    - action: 수행된 행동 (Tensor)

    현재는 두 상태 간의 차이의 크기를 보상으로 사용합니다.
    """
    # 상태 차이 계산
    state_diff = (next_state - state).abs().mean()
    
    # 보상 계산 (상태 차이가 클수록 보상이 높아짐)
    reward = -state_diff

    return reward.item()
