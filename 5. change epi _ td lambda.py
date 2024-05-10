#change epi.py

import numpy as np
import matplotlib.pyplot as plt


class Env:
    def __init__(self):
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
        # 선택된 행동을 70% 확률로, 다른 행동을 30% 확률로 수행
        if np.random.rand() < 0.7:
            self.move(action)
        else:
            other_actions = [a for a in self.action_space if a != action]
            self.move(np.random.choice(other_actions))

        # 보상과 게임 종료 여부를 반환
        reward = self.get_reward(self.agent_pos)
        done = self.agent_pos == self.home_pos
        return reward, self.agent_pos, done

    def move(self, action):
        # 주어진 행동에 따라 에이전트의 위치를 업데이트
        if action == 0:  # up
            self.agent_pos['y'] = max(self.agent_pos['y'] - 1, 0)
        elif action == 1:  # right
            self.agent_pos['x'] = min(self.agent_pos['x'] + 1, self.x_max)
        elif action == 2:  # down
            self.agent_pos['y'] = min(self.agent_pos['y'] + 1, self.y_max)
        elif action == 3:  # left
            self.agent_pos['x'] = max(self.agent_pos['x'] - 1, 0)

    def get_reward(self, pos):
        # 현재 위치에 따른 보상을 반환
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
def td_lambda(env, policy, lam, episodes, alpha=0.1, gamma=0.9):
    # 가치 벡터 초기화 및 플로팅을 위한 빈 리스트 생성
    value_vector = np.zeros(len(env.state_space))
    plot_x, plot_y = [], []

    # 에피소드 반복
    for episode in range(episodes):
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

        # 플로팅 업데이트
        if episode % (episodes // 100) == 0:
            plot_x.append(episode)
            plot_y.append(np.mean(value_vector))

    # 플로팅 데이터 반환
    return plot_x, plot_y


# 환경과 정책 초기화
env = Env()
policy = [np.full(len(env.action_space), 1.0 / len(env.action_space)) for _ in env.state_space]

# TD-Lambda 실행 및 그래프 생성
plt.figure(figsize=(12, 8))
episode_counts = [500, 1000, 3000, 5000]
lam = 0.3  # 람다값 고정
for episodes in episode_counts:
    plot_x, plot_y = td_lambda(env, policy, lam, episodes)
    plt.plot(plot_x, plot_y, label=f"Episodes={episodes}")

# 그래프 제목, 레이블 및 범례 설정, 그리고 그래프 표시
plt.title("TD-Lambda Performance Over Different Episode Counts")
plt.xlabel("Episode")
plt.ylabel("Average of value estimates")
plt.legend()
plt.show()
