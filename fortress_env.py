# 필요한 라이브러리 임포트
import gym
import numpy as np
import pyautogui
import cv2
from gym import spaces
from Action_List import actions, executer
from screen_capture import get_processed_screen  # ✅ 외부 화면 캡처 함수 사용 예정

# 강화학습용 환경 클래스 정의 (Gym 기반)
class FortressEnv(gym.Env):
    def __init__(self):
        super(FortressEnv, self).__init__()

        # 관측(observation)의 형태: 96x96 RGB 이미지
        self.obs_shape = (96, 96, 3)

        # 관측 공간 정의 (픽셀 값: 0~255)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)

        # 행동 공간 정의: [click_in, click_out, x좌표, y좌표] (모두 연속값 0~1)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    # 환경 초기화 시 호출되는 함수 (에피소드 시작)
    def reset(self):
        obs = self._get_screen()  # 초기 상태로 현재 화면 캡처
        return obs

    # 한 스텝 수행 (행동 → 상태, 보상, 종료여부 반환)
    def step(self, action):
        self._execute_action(action)           # 행동 실행 (마우스 조작)
        next_obs = self._get_screen()          # 다음 상태 관측 (화면 캡처)
        reward = 0.0                           # 외부 보상 없음 (ICM이 줄 예정)
        done = False                           # 종료 조건은 아직 미정
        return next_obs, reward, done, {}      # info는 비워둠

    # 화면 캡처는 외부 모듈에서 처리
    def _get_screen(self):
        return get_processed_screen()  # ✅ 외부에서 캡처 및 리사이즈하여 반환됨

    # 행동 벡터를 받아서 마우스 제어 동작 실행
    def _execute_action(self, action_vec):
        act = actions(*action_vec)       # 행동 클래스에 값 세팅
        exe = executer(act)              # 실행기 클래스 생성
        exe.execute()                    # 실제 마우스 조작 실행
