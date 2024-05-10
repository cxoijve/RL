import numpy as np  # Numpy 라이브러리를 np라는 이름으로 가져옴 (다차원 배열 관련 기능 제공)
import pygame  # Pygame 라이브러리 가져옴 (게임 개발에 사용)
import sys  # 시스템 관련 기능을 위한 라이브러리 가져옴

# Pygame 초기화
pygame.init()

# 화면 및 게임 관련 상수 정의
SCREEN_WIDTH = 400  # 화면의 너비를 400 픽셀로 설정
SCREEN_HEIGHT = 400  # 화면의 높이를 400 픽셀로 설정
GRID_SIZE = 4  # 격자의 크기를 4x4로 설정
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE  # 각 격자 셀의 크기 계산 (화면 너비를 격자 크기로 나눔)
AGENT_COLOR = (255, 0, 0)  # 에이전트의 색상을 빨간색으로 설정 (RGB)
SCHOOL_COLOR = (0, 255, 0)  # 학교의 색상을 초록색으로 설정 (RGB)
HOME_COLOR = (255, 255, 0)  # 집의 색상을 노란색으로 설정 (RGB)
BLUE_COLOR = (0, 0, 128)  # 특별 영역의 색상을 어두운 파란색으로 설정 (RGB)
BACKGROUND_COLOR = (255, 255, 255)  # 배경색을 흰색으로 설정 (RGB)
BLACK = (0, 0, 0)  # 검정색 설정 (RGB)

# Pygame 화면 설정
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))  # 화면 크기 설정
pygame.display.set_caption("WayBackHome")  # 게임 창의 제목 설정

# 폰트 설정
font = pygame.font.Font(None, 36)  # 기본 폰트와 크기 36으로 폰트 객체 생성

class Env:
    def __init__(self):
        # 초기 위치 및 게임 환경 설정
        self.agent_pos = {'y': 0, 'x': 0}  # 에이전트의 시작 위치 (0,0)
        self.school_pos = {'y': 0, 'x': 3}  # 학교의 위치 (0,3)
        self.home_pos = {'y': 3, 'x': 3}  # 집의 위치 (3,3)
        self.blue_area = {'y': [1, 2], 'x': [1, 2]}  # 특별 영역 (파란색 영역)의 위치
        self.y_min, self.x_min, self.y_max, self.x_max = 0, 0, 3, 3  # 격자의 최소 및 최대 y, x 값
        self.action_space = [0, 1, 2, 3]  # 가능한 행동들 (0: 상, 1: 우, 2: 하, 3: 좌)
        self.reset()  # 게임 환경 초기화

    def set_state(self):
        # 현재 게임 상태(격자) 설정
        state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=object)  # GRID_SIZE x GRID_SIZE 크기의 격자 초기화
        # 각 위치에 해당하는 객체 배치
        state[self.school_pos['y'], self.school_pos['x']] = 'S'  # 학교 위치
        state[self.home_pos['y'], self.home_pos['x']] = 'H'  # 집 위치
        for y in self.blue_area['y']:  # 파란 영역 설정
            for x in self.blue_area['x']:
                state[y, x] = 'B'
        state[self.agent_pos['y'], self.agent_pos['x']] = 'A'  # 에이전트 위치
        return state

    def reset(self):
        # 게임 상태 초기화
        self.agent_pos = {'y': 0, 'x': 0}  # 에이전트 위치 초기화
        self.state = self.set_state()  # 게임 상태 설정

    def reward(self, prev_state, action):
        # 보상 계산 함수
        reward = -0.5  # 기본 보상 설정
        done = False  # 게임 종료 여부

        y, x = self.agent_pos['y'], self.agent_pos['x']  # 현재 에이전트 위치

        # 위치에 따른 보상 조정
        if self.agent_pos == self.school_pos:
            reward += 50  # 학교에 도착 시 보상
        elif self.agent_pos == self.home_pos:
            reward += 100  # 집에 도착 시 보상, 게임 종료
            done = True
        elif prev_state[y, x] == 'B':
            reward -= 10  # 파란 영역에 들어가면 패널티

        return reward, done

    def step(self, action):
        # 에이전트의 행동 처리
        if np.random.rand() < 0.3:  # 30% 확률로 무작위 행동 선택
            action = np.random.choice(self.action_space)

        # 에이전트 위치 업데이트
        if action == 0:  # 위로 이동
            self.agent_pos['y'] = max(self.agent_pos['y'] - 1, self.y_min)
        elif action == 1:  # 오른쪽으로 이동
            self.agent_pos['x'] = min(self.agent_pos['x'] + 1, self.x_max)
        elif action == 2:  # 아래로 이동
            self.agent_pos['y'] = min(self.agent_pos['y'] + 1, self.y_max)
        elif action == 3:  # 왼쪽으로 이동
            self.agent_pos['x'] = max(self.agent_pos['x'] - 1, self.x_min)

        prev_state = self.state.copy()  # 이전 상태 복사
        self.state = self.set_state()  # 새 상태 설정
        reward, done = self.reward(prev_state, action)  # 보상 및 게임 종료 여부 계산

        print(f"Action: {action}, Reward: {reward}")  # 행동과 보상 출력

        return reward, done

# 게임 환경 객체 생성
env = Env()
total_reward = 0  # 총 보상 초기화
running = True  # 게임 실행 여부

# 게임 루프
while running:
    for event in pygame.event.get():  # 이벤트 처리
        if event.type == pygame.QUIT:  # 게임 종료 이벤트
            running = False

    action = np.random.choice(env.action_space)  # 무작위 행동 선택
    reward, done = env.step(action)  # 행동 수행 및 보상, 게임 종료 여부 반환
    total_reward += reward  # 총 보상 업데이트

    screen.fill(BACKGROUND_COLOR)  # 화면 배경색으로 채우기

    # 격자선 그리기
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))

    # 게임 상태에 따라 셀 그리기
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            cell_value = env.state[y, x]
            if cell_value == 'A':
                pygame.draw.rect(screen, AGENT_COLOR, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif cell_value == 'S':
                pygame.draw.rect(screen, SCHOOL_COLOR, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif cell_value == 'H':
                pygame.draw.rect(screen, HOME_COLOR, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif cell_value == 'B':
                pygame.draw.rect(screen, BLUE_COLOR, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    reward_text = font.render(f"Total Reward: {total_reward}", True, BLACK)  # 총 보상 표시 문자열 생성
    screen.blit(reward_text, (10, 10))  # 총 보상을 표시하는 위치를 (10, 10)으로 설정한 것은 화면의 왼쪽 상단 모서리 근처를 의미

    # 게임 종료 시 처리
    if done:
        running = False  # 게임 실행 상태를 False로 변경
        pygame.time.wait(2000)  # 2000밀리초(2초) 동안 대기

    pygame.display.flip()  # 화면을 업데이트하여 변경사항 반영
    pygame.time.delay(100)  # 100밀리초 동안 대기 (게임의 프레임 속도 조절)

# 게임 루프 종료 후 처리
pygame.quit()  # Pygame 종료
sys.exit()  # 프로그램 종료