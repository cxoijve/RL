import numpy as np
import matplotlib.pyplot as plt

gamma = 0.9
alpha_init = 3e-1
k_alpha = 1e-1

def get_state_index(state_space, state):
    for i_s, s in enumerate(state_space):
        if np.array_equal(s, state):
            return i_s
    return 0  # Couldn't find the state, return 0 (or any other default value)

def calc_return(gamma, rewards):
    n = len(rewards)
    rewards = np.array(rewards)
    gammas = gamma * np.ones([n])
    powers = np.arange(n)
    power_of_gammas = np.power(gammas, powers)
    discounted_rewards = rewards * power_of_gammas
    g = np.sum(discounted_rewards)
    return g

# n-step TD 가치 예측을 수행하는 함수
def n_step_td_value_prediction(env, policy, n, num_iterations):
    # 각 상태의 가치를 저장할 배열을 초기화
    value_vector = np.zeros([len(env.state_space)])
    # 결과를 저장할 버퍼를 초기화
    plot_buffer = {'x': [], 'y': []}

    # 주어진 반복 횟수만큼 학습을 수행
    for loop_count in range(num_iterations):
        # 상태, 행동, 보상을 추적하는 경로를 초기화
        trajectory = {'states': [], 'actions': [], 'rewards': []}
        done = False # 에피소드 종료 플래그
        step_count = 0
        state = env.reset()  # 환경 초기화 및 현재 상태 설정

        # 에피소드가 종료될 때까지 반복
        while not done:
            i_s = get_state_index(env.state_space, state) # 현재 상태의 인덱스 찾기
            pi_s = policy[i_s]
            a = np.random.choice(env.action_space, p=pi_s)
            next_state, reward, done = env.step(a)  # 행동 수행 및 결과 얻기

            # 경로에 상태, 행동, 보상을 추가
            next_state = env.current_state
            trajectory['states'].append(state)
            trajectory['actions'].append(a)
            trajectory['rewards'].append(reward)
            # 다음 상태로 업데이트
            state = next_state
            step_count += 1

            # n+1보다 많은 단계가 저장되면 오래된 단계를 제거
            if step_count >= n + 1:
                trajectory['states'].pop(0)
                trajectory['actions'].pop(0)
                trajectory['rewards'].pop(0)

            # n단계가 충분히 쌓이면 가치 업데이트를 수행
            if step_count >= n:
                s_t_sub_n = trajectory['states'][0]
                i_s_t_sub_n = get_state_index(env.state_space, s_t_sub_n)
                s_t = trajectory['states'][-1]
                i_s_t = get_state_index(env.state_space, s_t)
                # 가치 업데이트를 수행
                alpha = alpha_init / (1 + k_alpha * loop_count)
                discounted_rewards = calc_return(gamma, trajectory['rewards'])
                td = discounted_rewards + (gamma ** n) * value_vector[i_s_t] - value_vector[i_s_t_sub_n]
                value_vector[i_s_t_sub_n] += alpha * td
        # 주기적으로 결과를 저장
        if (loop_count + 1) % 100 == 0:
            plot_buffer['x'].append(loop_count)
            plot_buffer['y'].append(value_vector[0])

    return plot_buffer

 # 4x4 격자의 상태 공간과 행동 공간을 정의
class WayBackHomeEnv:
    def __init__(self):
        self.state_space = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.action_space = [0, 1, 2, 3] # 행동: 0=상, 1=우, 2=하, 3=좌
        self.current_state = None # 현재 상태 저장 변수

        # 에이전트의 위치, 학교와 집의 위치, 그리고 파란 영역의 범위
        self.agent_pos = {'y': 0, 'x': 0}
        self.school_pos = {'y': 0, 'x': 3}
        self.home_pos = {'y': 3, 'x': 3}
        self.blue_area = {'y': [1, 2], 'x': [1, 2]}
        # 격자의 최소 및 최대 좌표값
        self.y_min, self.x_min, self.y_max, self.x_max = 0, 0, 3, 3
        self.reset()

    def reset(self):
        # 에이전트 위치를 초기화하고 상태를 설정
        self.agent_pos = {'y': 0, 'x': 0}
        self.current_state = self.set_state() # 초기 상태 반환

    def set_state(self):
        # 격자 상태를 초기화하고 에이전트, 학교, 집, 파란 영역의 위치를 설정
        state = np.zeros((4, 4), dtype=object)
        state[self.school_pos['y'], self.school_pos['x']] = 'S'
        state[self.home_pos['y'], self.home_pos['x']] = 'H'
        for y in self.blue_area['y']:
            for x in self.blue_area['x']:
                state[y, x] = 'B'
        state[self.agent_pos['y'], self.agent_pos['x']] = 'A'
        return state  # 현재 상태 반환

    def reward(self, prev_state, action):
        # 보상과 게임 종료 여부를 계산
        reward = -0.5 # 기본적으로 -0.5의 보상을 설정
        done = False # 게임 종료 여부 초기값

        # 에이전트 위치 확인
        y, x = self.agent_pos['y'], self.agent_pos['x']

        # 학교, 집, 파란 영역에 따라 보상을 조정
        if self.agent_pos == self.school_pos:
            reward += 50 # 학교에 도착하면 보상
        elif self.agent_pos == self.home_pos:
            reward += 100 # 집에 도착하면 보상과 게임 종료
            done = True
        elif prev_state[y, x] == 'B':
            reward -= 10 # 파란 영역에 도달하면 패널티

        return reward, done

    def step(self, action):
        # 주어진 행동에 따라 에이전트의 상태를 업데이트하고 결과를 반환
        if np.random.rand() < 0.3:
            action = np.random.choice(self.action_space) # 30% 확률로 무작위 행동 선택

        # 에이전트의 위치 업데이트
        y, x = self.agent_pos['y'], self.agent_pos['x']
        if action == 0 and y > self.y_min:
            y -= 1  # 위로 이동
        elif action == 1 and x < self.x_max:
            x += 1  # 오른쪽으로 이동
        elif action == 2 and y < self.y_max:
            y += 1  # 아래로 이동
        elif action == 3 and x > self.x_min:
            x -= 1  # 왼쪽으로 이동

        self.agent_pos = {'y': y, 'x': x}  # 업데이트된 위치를 저장
        self.current_state = self.set_state()  # 현재 상태 업데이트
        reward, done = self.reward(self.current_state, action) # 보상 계산

        return (y, x), reward, done  # 업데이트된 위치와 보상, 게임 종료 여부 반환


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    env = WayBackHomeEnv()
    # 상태 공간을 4x4로 업데이트
    env.state_space = [(i, j) for i in range(4) for j in range(4)]
    # 정책 배열을 상태 공간의 크기에 맞게 업데이트
    policy = np.full((len(env.state_space), len(env.action_space)), 0.25)

    n_values = [1, 3, 5]
    colors = ['red', 'green', 'blue']
    num_iterations = 5000  # 반복 횟수 증가
    alpha_init = 0.1  # 초기 학습률
    k_alpha = 0.01    # 학습률 감소율 조정

    plt.figure(figsize=(10, 6))
    # n-step TD 가치 예측을 수행하고 결과를 그래프에 표시
    for n, color in zip(n_values, colors):
        plot_buffer = n_step_td_value_prediction(env, policy, n, num_iterations)
        plt.plot(plot_buffer['x'], plot_buffer['y'], label=f'n={n}', color=color)

    plt.xlabel('Iterations')
    plt.ylabel('Value Estimation')
    plt.title('n-step TD Value Estimations')
    plt.legend()
    plt.show()