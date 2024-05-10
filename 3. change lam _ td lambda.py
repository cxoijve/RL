#change lambda.py

import numpy as np
import matplotlib.pyplot as plt

class Env: # 환경 클래스 정의
    def __init__(self):
        # 액션 및 상태 공간, 최대 y 및 x 좌표, 홈 및 특별 위치, 파란색 영역 정의
        self.action_space = [0, 1, 2, 3]  # 상, 하, 좌, 우
        self.state_space = [{'y': y, 'x': x} for y in range(4) for x in range(4)]
        self.y_max, self.x_max = 3, 3
        self.home_pos = {'y': 3, 'x': 3}  # 'H' 위치
        self.special_pos = {'y': 0, 'x': 1}  # 'S' 위치
        self.blue_zones = [{'y': 2, 'x': 2}, {'y': 2, 'x': 3}]  # 파란색 영역 위치

    def reset(self):
        # 에이전트의 위치를 시작 위치로 재설정
        self.agent_pos = {'y': 0, 'x': 0}
        return self.agent_pos

    def step(self, action):
        # 선택된 행동을 70% 확률로, 다른 무작위 행동을 30% 확률로 수행
        if np.random.rand() < 0.7:
            self.move(action)
        else:
            other_actions = [a for a in self.action_space if a != action]
            self.move(np.random.choice(other_actions))
        # 보상 및 게임 종료 여부 반환
        reward = self.get_reward(self.agent_pos)
        done = self.agent_pos == self.home_pos
        return reward, self.agent_pos, done

    def move(self, action):
        # 선택된 행동에 따라 에이전트의 위치 업데이트
        if action == 0:  # 위로
            self.agent_pos['y'] = max(self.agent_pos['y'] - 1, 0)
        elif action == 1:  # 오른쪽으로
            self.agent_pos['x'] = min(self.agent_pos['x'] + 1, self.x_max)
        elif action == 2:  # 아래로
            self.agent_pos['y'] = min(self.agent_pos['y'] + 1, self.y_max)
        elif action == 3:  # 왼쪽으로
            self.agent_pos['x'] = max(self.agent_pos['x'] - 1, 0)

    def get_reward(self, pos):
        # 현재 위치에 따른 보상 반환
        if pos == self.special_pos:
            return 50
        elif pos == self.home_pos:
            return 100
        elif pos in self.blue_zones:
            return -10
        else:
            return -0.5

# 상태를 상태 공간에서의 인덱스로 변환하는 함수 정의
def get_state_index(state_space, state):
    return state_space.index(state)

# TD(lambda) 함수 정의
def td_lambda(env, policy, lam, color, alpha=0.1, gamma=0.9):
    value_vector = np.zeros(len(env.state_space))  # 가치 함수 추정치 저장
    plot_x, plot_y = [], []

    # 에피소드 반복
    for episode in range(1500):
        # 환경 재설정 및 초기 상태 및 상태 인덱스 가져오기
        state = env.reset()
        state_index = get_state_index(env.state_space, state)
        eligibility_trace = np.zeros(len(env.state_space))

        # 에피소드 루프
        while True:
            # 정책에 기반하여 행동 선택
            action = np.random.choice(env.action_space, p=policy[state_index])
            # 환경에서 한 단계 전진 및 다음 상태, 보상 및 완료 상태 가져오기
            reward, next_state, done = env.step(action)
            next_state_index = get_state_index(env.state_space, next_state)

            # TD 오류와 추적(trace) 업데이트
            td_error = reward + gamma * value_vector[next_state_index] - value_vector[state_index]
            eligibility_trace[state_index] += 1

            # 모든 상태에 대해 가치 함수 업데이트
            for s in range(len(env.state_space)):
                value_vector[s] += alpha * td_error * eligibility_trace[s]
                eligibility_trace[s] *= gamma * lam

            # 에피소드 종료 시 중단
            if done:
                break
            # 현재 상태 및 상태 인덱스 업데이트
            state = next_state
            state_index = next_state_index

        if episode % 10 == 0:
            plot_x.append(episode)
            plot_y.append(np.mean(value_vector))

    plt.plot(plot_x, plot_y, color=color, label=f"lambda={lam}")

# 환경 및 정책 초기화
env = Env()
policy = [np.full(len(env.action_space), 1.0 / len(env.action_space)) for _ in env.state_space]

# 그래프 설정 및 TD-Lambda 실행
plt.figure(figsize=(12, 8))
lam_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 람다 값 변경
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']  # 색상 변경
for lam, color in zip(lam_values, colors):
    td_lambda(env, policy, lam, color)

plt.title("Estimated TD-Lambda value function over time with different lambda values")
plt.xlabel("Episode")
plt.ylabel("Average of value estimates")
plt.legend()
plt.ylim(bottom=-5)
plt.show()
