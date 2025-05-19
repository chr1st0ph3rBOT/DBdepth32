# icm_module.py
import torch
import torch.nn as nn

class ICM(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=256):
        """
        obs_shape: 관측 상태의 이미지 크기 (예: (3, 96, 96))
        action_dim: 행동 벡터 차원 수 (예: 4)
        feature_dim: CNN으로 추출할 상태 임베딩 크기 (기본 256)
        """
        super(ICM, self).__init__()
        
        c, h, w = obs_shape  # 채널, 높이, 너비

        # 🧠 상태 인코더 (이미지를 임베딩 벡터로 압축)
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),  # 96 → 23 kernel은 필터임임
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 23 → 10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 10 → 8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, feature_dim),  # CNN 출력 → FC → feature_dim 크기
            nn.ReLU()
        )

        # 🔮 Forward Model: (현재 상태, 행동) → 다음 상태 예측
        self.forward = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward_loss(self, state, next_state, action):
        """
        상태 전이 예측 기반 intrinsic reward 계산
        - 입력: 현재 상태, 다음 상태, 행동
        - 출력: 예측 실패 정도 (보상 값으로 사용)
        """

        # ▶️ 다음 상태를 인코딩 (정답 역할)
        with torch.no_grad():  # 타겟은 학습하지 않음
            phi_next = self.encoder(next_state)

        # ▶️ 현재 상태도 인코딩
        phi = self.encoder(state)

        # ▶️ 상태 임베딩 + 행동 결합 → 다음 상태 예측
        input_fwd = torch.cat([phi, action], dim=1)
        phi_next_pred = self.forward(input_fwd)

        # ▶️ 예측과 실제의 차이 = 보상 (예측이 어려울수록 보상이 큼)
        loss = ((phi_next - phi_next_pred) ** 2).mean(dim=1)

        return loss  # shape: [batch_size] (각 transition 당 보상)
