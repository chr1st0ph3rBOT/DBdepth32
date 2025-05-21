# modules/reward_module.py

class RewardModule:
    def __init__(self, hp_weight=1.5, item_weight=3.0, time_weight=0.2):
        self.hp_weight = hp_weight
        self.item_weight = item_weight
        self.time_weight = time_weight

    def external_reward_func(self, my_hp_change, enemy_hp_change, item_count, time_seconds):
        """
        외부 보상 계산 (튜닝된 가중치 적용)
        """
        hp_score = self.hp_weight * (my_hp_change - enemy_hp_change)
        item_score = self.item_weight * item_count
        time_score = -self.time_weight * time_seconds  # 오래 걸릴수록 패널티

        reward = hp_score + item_score + time_score
        return reward
