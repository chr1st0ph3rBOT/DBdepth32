import torch
from icm_module import ICM


def calculate_icm_reward(icm_model, state, next_state, action):
    """
    ICM 모듈을 기반으로 보상을 계산하는 함수.
    - icm_model: ICM 모델 인스턴스
    - state: 현재 상태 (Tensor)
    - next_state: 다음 상태 (Tensor)
    - action: 수행된 행동 (Tensor)
    
    ICM의 예측 오류를 보상으로 반환합니다.
    """
    icm_model.eval()
    with torch.no_grad():
        # 예측 오류 계산
        reward = icm_model.forward_loss(state, next_state, action)

    return reward.mean().item()
