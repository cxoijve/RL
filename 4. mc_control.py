import numpy as np
import random

# 상수 정의
GRID_SIZE = 4

class WayBackHomeEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.action_space = ['up', 'down', 'left', 'right']
        self.state_space = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.home_pos = (3, 3)
        self.special_pos = (0, 1)
        self.blue_zone = [(1, 1), (1, 2), (2, 1), (2, 2)]
        self.agent_pos = (0, 0)
        self.total_reward = 0

    def reset(self):
        """환경을 초기화하고 에이전트의 위치를 리셋하며 초기 상태 반환"""
        self.agent_pos = (0, 0)
        self.total_reward = 0
        return self.agent_pos

    def step(self, action):
        """주어진 액션에 따라 에이전트를 한 단계 전진시키고 보상 및 종료 여부 반환"""
        if random.random() > 0.7:
            action = random.choice(self.action_space)

        y, x = self.agent_pos
        if action == 'up' and y > 0:
            y -= 1
        elif action == 'down' and y < self.grid_size - 1:
            y += 1
        elif action == 'left' and x > 0:
            x -= 1
        elif action == 'right' and x < self.grid_size - 1:
            x += 1

        self.agent_pos = (y, x)
        reward = -0.5
        if self.agent_pos in self.blue_zone:
            reward -= 10
        elif self.agent_pos == self.special_pos:
            reward += 50
        elif self.agent_pos == self.home_pos:
            reward += 100

        self.total_reward += reward

        return self.agent_pos, reward, self.agent_pos == self.home_pos

# MC Control 함수 정의
def mc_control(env, num_episodes=5000):
    gamma = 0.9
    eps = 0.05

    def get_state_index(state_space, state):
        """주어진 상태가 상태 공간에서 몇 번째 인덱스인지 반환"""
        for i_s, s in enumerate(state_space):
            if s == state:
                return i_s
        assert False, "Couldn't find the state from the state space"

    def calc_return(gamma, rewards):
        """할인된 총 보상(G)을 계산하여 반환"""
        n = len(rewards)
        rewards = np.array(rewards)
        gammas = gamma * np.ones([n])
        powers = np.arange(n)
        power_of_gammas = np.power(gammas, powers)
        discounted_rewards = rewards * power_of_gammas
        g = np.sum(discounted_rewards)
        return g

    def mc_control_es(env, num_episodes=10000):
        """Exploring Starts를 사용한 MC Control 알고리즘"""
        Q = np.zeros((len(env.state_space), len(env.action_space)))
        N = np.zeros((len(env.state_space), len(env.action_space)))
        policy = np.ones((len(env.state_space), len(env.action_space))) / len(env.action_space)  # 초기 정책 설정

        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []

            state = env.reset()
            done = False

            while not done:
                action = np.random.choice(env.action_space)
                next_state, reward, done = env.step(action)
                states.append(tuple(state))
                actions.append(env.action_space.index(action))
                rewards.append(reward)
                state = next_state

            G = 0
            for t in reversed(range(len(states))):
                G = gamma * G + rewards[t]
                s = states[t]
                a = actions[t]
                if (s, a) not in zip(states[:t], actions[:t]):
                    N[s, a] += 1
                    action_probabilities = np.array([eps / len(env.action_space)] * len(env.action_space))
                    action_probabilities[np.argmax(Q[s])] = 1 - eps + eps / len(env.action_space)
                    policy[s] = action_probabilities  # policy 배열에 직접 할당

        return policy, Q

    def mc_control_epsilon_soft(env, policy, num_episodes=20000):
        """Epsilon-Greedy를 사용한 MC Control 알고리즘"""
        # 각 상태-액션 쌍에 대한 Q값과 방문 횟수를 초기화
        Q = np.zeros((len(env.state_space), len(env.action_space)))
        N = np.zeros((len(env.state_space), len(env.action_space)))

        # 에피소드 수 만큼 루프 실행
        for episode in range(num_episodes):
            # 현재 에피소드의 상태, 액션, 보상을 저장하기 위한 리스트
            states = []
            actions = []
            rewards = []

            # 환경을 리셋하여 새로운 에피소드 시작
            state = env.reset()
            done = False

            # 종료 조건이 될 때까지 에피소드 실행
            while not done:
                if random.random() < 1 - eps:
                    action = np.random.choice(env.action_space, p=policy[get_state_index(env.state_space, state)])
                else:
                    action = np.random.choice(env.action_space)
                next_state, reward, done = env.step(action)
                states.append(tuple(state))
                actions.append(env.action_space.index(action))
                rewards.append(reward)
                state = next_state

            G = 0
            for t in reversed(range(len(states))):
                G = gamma * G + rewards[t]
                s = states[t]
                a = actions[t]
                if (s, a) not in zip(states[:t], actions[:t]):
                    N[s, a] += 1
                    Q[s, a] += (G - Q[s, a]) / N[s, a]
                    policy[s] = np.eye(len(env.action_space))[np.argmax(Q[s])]  # policy 배열에 직접 할당

        return policy, Q


        # 소수점 3자리까지 출력
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        # 환경 초기화
        env = WayBackHomeEnv()

        # MC Control with Exploring Starts를 사용하여 정책 학습
        # 초기 정책을 모든 액션에 대해 균등하게 설정
        policy = np.ones((len(env.state_space), len(env.action_space))) / len(env.action_space)  # 수정: 초기 정책 설정
        # MC Control with Exploring Starts 함수 호출
        policy_es, action_value_matrix_es = mc_control_es(env, num_episodes=10000)

        # MC Control with Epsilon-Greedy를 사용하여 정책 학습
        # 초기 정책을 모든 액션에 대해 epsilon-soft로 설정
        policy = np.ones((len(env.state_space), len(env.action_space))) * eps / len(env.action_space)  # 수정: 초기 정책 설정
        # MC Control with Epsilon-Greedy 함수 호출
        policy_eps_soft, action_value_matrix_eps_soft = mc_control_epsilon_soft(env, num_episodes=20000)

        # Exploring Starts로 학습한 정책의 가치 함수 계산
        value_vector_es = np.sum(policy_es * action_value_matrix_es, axis=-1)
        value_table_es = value_vector_es.reshape(4, 4)

        # Epsilon-Greedy로 학습한 정책의 가치 함수 계산
        value_vector_eps_soft = np.sum(policy_eps_soft * action_value_matrix_eps_soft, axis=-1)
        value_table_eps_soft = value_vector_eps_soft.reshape(4, 4)

        # 결과 출력
        print("Value Function (MC Control with Exploring Starts):")
        for row in value_table_es:
            print(" ".join(f"{value: .3f}" for value in row))

        print("\nValue Function (MC Control with Epsilon-Greedy):")
        for row in value_table_eps_soft:
            print(" ".join(f"{value: .3f}" for value in row))

if __name__ == "__main__":
    # 환경 초기화
    env = WayBackHomeEnv()
    # MC Control 알고리즘 실행
    mc_control(env, num_episodes=5000)