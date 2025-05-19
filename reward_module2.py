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


def define_ex_reward():
    sdhfsdfbdjsf

    return result

# 최종 보상 = 외재 보상 + curiosity 보상 (weight 조절 가능)
result_reward = extrinsic_reward + beta * curiosity_reward
