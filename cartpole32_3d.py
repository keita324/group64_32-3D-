import gym
import csv
import os
import numpy as np
import time
import datetime 
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import tensorflow as tf
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ==========================================================
# 定数の定義
# ==========================================================
ACTOR_EPSILON = 0.2          # アクター選択のランダム性を制御する確率
RECENT_EPISODES_WINDOW = 5   # 直近のエピソード数を考慮する窓サイズ
NUM_EPISODES = 100          # 学習を行う最大エピソード数
EVALUATION_POINT = 50       # 中間評価を行うエピソード数
MAX_STEPS = 200            # 1エピソードの最大ステップ数
BATCH_SIZE = 32            # 学習時のバッチサイズ
MEMORY_SIZE = 10000        # 経験replay用のメモリサイズ
LEARNING_RATE = 0.0001     # 学習率
GAMMA = 0.9                # 割引率

# 乱数シードの設定（実験の再現性のため）
seed = 5
np.random.seed(seed)
tf.random.set_seed(seed)

# ==========================================================
# データ構造の定義
# ==========================================================
class EvaluationData:
    """評価データを保持するクラス"""
    def __init__(self, actor_name, evaluation_type, episode_num):
        self.actor_name = actor_name            # アクター名（'mario' or 'luigi'）
        self.evaluation_type = evaluation_type  # 評価タイプ（'mid' or 'final'）
        self.episode_num = episode_num         # エピソード番号
        self.steps = 0                         # 実行ステップ数
        self.total_reward = 0                  # 累積報酬
        self.average_state_values = []         # 状態値の平均
        self.max_state_values = []             # 状態値の最大
        self.min_state_values = []             # 状態値の最小
        self.action_distribution = {0: 0, 1: 0} # 行動の分布
        self.time_taken = 0                    # 実行時間

# ==========================================================
# Q値可視化関数（新規追加）
# ==========================================================
def visualize_q_values(mario_qn, luigi_qn, episode_num, save_dir):
    """
    マリオとルイージのQ値を3次元グラフで可視化
    
    Args:
        mario_qn: マリオのQネットワーク
        luigi_qn: ルイージのQネットワーク
        episode_num: エピソード番号
        save_dir: 保存ディレクトリ
    """
    # CartPoleの状態空間の範囲
    cart_positions = np.linspace(-2.4, 2.4, 40)
    pole_angles = np.linspace(-0.21, 0.21, 40)  # 約±12度
    
    # 固定する他の状態変数
    cart_velocity = 0.0
    pole_velocity = 0.0
    
    X, Y = np.meshgrid(cart_positions, pole_angles)
    
    # 各エージェント用のfigure
    agents = [('Mario_A64', mario_qn), ('Luigi_A32', luigi_qn)]
    
    for agent_name, qn in agents:
        Z_left = np.zeros_like(X)   # 左に動く場合のQ値
        Z_right = np.zeros_like(X)  # 右に動く場合のQ値
        
        # 各グリッドポイントでQ値を計算
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([[X[i,j], cart_velocity, Y[i,j], pole_velocity]])
                q_values = qn.model.predict(state, verbose=0)[0]
                Z_left[i,j] = q_values[0]   # 行動0（左）のQ値
                Z_right[i,j] = q_values[1]  # 行動1（右）のQ値
        
        # 最大Q値を取る
        Z_max = np.maximum(Z_left, Z_right)
        
        # 3次元プロット
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # サーフェスプロット
        surf = ax.plot_surface(X, Y, Z_max, cmap=cm.coolwarm, 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # グラフの設定
        ax.set_xlabel('Cart Position', fontsize=12)
        ax.set_ylabel('Pole Angle (rad)', fontsize=12)
        ax.set_zlabel('Q-value', fontsize=12)
        ax.set_title(f'{agent_name} Q-values at Episode {episode_num}', fontsize=14)
        
        # カラーバーの追加
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # 視点の設定
        ax.view_init(elev=30, azim=45)
        
        # 保存
        filename = os.path.join(save_dir, f'{agent_name}_qvalues_ep{episode_num}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def compare_q_values(mario_qn, luigi_qn, episode_num, save_dir):
    """
    マリオとルイージのQ値を比較可視化
    
    Args:
        mario_qn: マリオのQネットワーク
        luigi_qn: ルイージのQネットワーク
        episode_num: エピソード番号
        save_dir: 保存ディレクトリ
    """
    cart_positions = np.linspace(-2.4, 2.4, 30)
    pole_angles = np.linspace(-0.21, 0.21, 30)
    
    X, Y = np.meshgrid(cart_positions, pole_angles)
    
    fig = plt.figure(figsize=(15, 6))
    
    agents = [('Mario (A64)', mario_qn), ('Luigi (A32)', luigi_qn)]
    
    for idx, (agent_name, qn) in enumerate(agents):
        ax = fig.add_subplot(1, 2, idx+1, projection='3d')
        
        Z_max = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([[X[i,j], 0.0, Y[i,j], 0.0]])
                q_values = qn.model.predict(state, verbose=0)[0]
                Z_max[i,j] = np.max(q_values)
        
        surf = ax.plot_surface(X, Y, Z_max, cmap=cm.coolwarm, 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        ax.set_xlabel('Cart Position', fontsize=10)
        ax.set_ylabel('Pole Angle (rad)', fontsize=10)
        ax.set_zlabel('Q-value', fontsize=10)
        ax.set_title(f'{agent_name}', fontsize=12)
        ax.view_init(elev=30, azim=45)
    
    plt.suptitle(f'Q-value Comparison at Episode {episode_num}', fontsize=16)
    plt.tight_layout()
    
    # 保存
    filename = os.path.join(save_dir, f'qvalue_comparison_ep{episode_num}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================================
# CSV操作関連の関数
# ==========================================================
def safe_write_csv(filename, data, headers=None, mode='a'):
    """
    CSVファイルにデータを安全に書き込む
    
    Args:
        filename (str): 出力するCSVファイルのパス
        data (list): 書き込むデータ（単一行または複数行）
        headers (list, optional): CSVのヘッダー
        mode (str): ファイルオープンモード（'w'=新規作成, 'a'=追記）
    """
    with open(filename, mode, newline='') as f:
        writer = csv.writer(f)
        
        # 新規作成モードでヘッダーが指定されている場合、ヘッダーを書き込む
        if mode == 'w' and headers is not None:
            writer.writerow(headers)
        
        # データが空でない場合のみ書き込み処理を行う
        if data:
            # データを2次元リストに正規化
            if not isinstance(data[0], list):
                data = [data]
            
            # データを1行ずつ書き込む
            for row in data:
                writer.writerow(row)

def get_evaluation_headers():
    """評価データのCSVヘッダーを生成"""
    return [
        'actor_name',         # アクター名
        'evaluation_type',    # 評価タイプ
        'episode_num',        # エピソード番号
        'steps',              # ステップ数
        'total_reward',       # 累積報酬
        'avg_cart_position',  # カートの位置の平均
        'avg_cart_velocity',  # カートの速度の平均
        'avg_pole_angle',     # ポールの角度の平均
        'avg_pole_velocity',  # ポールの角速度の平均
        'max_cart_position',  # カートの位置の最大値
        'max_cart_velocity',  # カートの速度の最大値
        'max_pole_angle',     # ポールの角度の最大値
        'max_pole_velocity',  # ポールの角速度の最大値
        'min_cart_position',  # カートの位置の最小値
        'min_cart_velocity',  # カートの速度の最小値
        'min_pole_angle',     # ポールの角度の最小値
        'min_pole_velocity',  # ポールの角速度の最小値
        'action_0_count',     # 行動0（左）の選択回数
        'action_1_count',     # 行動1（右）の選択回数
        'time_taken'          # 実行時間
    ]

def format_evaluation_data(eval_data):
    """
    評価データをCSV形式の行データに変換
    
    Args:
        eval_data (EvaluationData): 評価データオブジェクト
    
    Returns:
        list: CSV行として整形されたデータ
    """
    return [
        eval_data.actor_name,
        eval_data.evaluation_type,
        eval_data.episode_num,
        eval_data.steps,
        eval_data.total_reward,
        *eval_data.average_state_values,  # 状態値の平均を展開
        *eval_data.max_state_values,      # 状態値の最大を展開
        *eval_data.min_state_values,      # 状態値の最小を展開
        eval_data.action_distribution[0],  # 行動0の回数
        eval_data.action_distribution[1],  # 行動1の回数
        eval_data.time_taken
    ]

# ==========================================================
# Q学習ネットワーク
# ==========================================================
class QNetwork:
    """Q学習のためのニューラルネットワーク"""
    def __init__(self, learning_rate=LEARNING_RATE, state_size=4, action_size=2, hidden_size=10):
        # ネットワークの構築
        self.model = Sequential([
            Dense(hidden_size, activation='relu', input_dim=state_size),
            Dense(hidden_size, activation='relu'),
            Dense(action_size, activation='linear')
        ])
        # オプティマイザと損失関数の設定
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)
    
    def replay(self, memory, batch_size, gamma, targetQN):
        """経験リプレイによる学習の実行"""
        # バッチサイズに応じた入力と目標値の配列を初期化
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))
        mini_batch = memory.sample(batch_size)
        
        # バッチ内の各サンプルについて学習を実行
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b
            
            # 次の状態が終端状態でない場合、Q値を計算
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                retmainQs = self.model.predict(next_state_b, verbose=0)[0]
                next_action = np.argmax(retmainQs)
                target = reward_b + gamma * targetQN.model.predict(next_state_b, verbose=0)[0][next_action]
            
            # 目標Q値の設定
            targets[i] = self.model.predict(state_b, verbose=0)
            targets[i][action_b] = target
        
        # ネットワークの更新
        self.model.fit(inputs, targets, epochs=1, verbose=0)

# ==========================================================
# メモリ管理
# ==========================================================
class Memory:
    """経験リプレイのためのメモリ管理クラス"""
    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        """新しい経験をメモリに追加"""
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """メモリからランダムにバッチサイズ分のサンプルを取得"""
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
    
    def len(self):
        """現在のメモリサイズを取得"""
        return len(self.buffer)

# ==========================================================
# アクター管理
# ==========================================================
class CollaborativeActor:
    """マリオとルイージの協調学習を管理するクラス"""
    def __init__(self):
        self.mario_steps = []    # マリオの実行ステップ履歴
        self.luigi_steps = []    # ルイージの実行ステップ履歴
        self.last_actor = None   # 直前のアクター
    
    def get_actor(self):
        """
        次のアクターを選択
        
        Returns:
            str: 選択されたアクター名（'mario' or 'luigi'）
        """
        if np.random.uniform(0, 1) > ACTOR_EPSILON:
            # 性能ベースの選択
            if self.last_actor is None:
                actor = np.random.choice(['mario', 'luigi'])
            elif len(self.mario_steps) == 0 or len(self.luigi_steps) == 0:
                actor = 'luigi' if self.last_actor == 'mario' else 'mario'
            else:
                # 直近の平均ステップ数に基づいて選択
                mario_avg = np.mean(self.mario_steps)
                luigi_avg = np.mean(self.luigi_steps)
                actor = 'mario' if mario_avg >= luigi_avg else 'luigi'
        else:
            # ランダム選択
            actor = np.random.choice(['mario', 'luigi'])
        
        self.last_actor = actor
        return actor

    def update_steps(self, actor, steps):
        """アクターの実行ステップ数を記録"""
        if actor == 'mario':
            self.mario_steps.append(steps)
        else:
            self.luigi_steps.append(steps)
    
    def get_action(self, state, mario_QN, luigi_QN, current_actor, episode, teach=False):
        """
        現在の状態に対する行動を選択
        
        Args:
            state: 現在の環境の状態
            mario_QN: マリオのQネットワーク
            luigi_QN: ルイージのQネットワーク
            current_actor: 現在のアクター名
            episode: 現在のエピソード番号
            teach: 教師フラグ（Trueの場合、常に最適行動を選択）
        
        Returns:
            int: 選択された行動（0 or 1）
        """
        # 現在のアクターのQネットワークを選択
        mainQN = mario_QN if current_actor == 'mario' else luigi_QN
        
        # イプシロンの計算（徐々に探索率を下げる）
        epsilon = 0.001 + 0.9 / (1.0 + episode)
        
        if np.random.uniform(0, 1) > epsilon or teach:
            # 最適行動の選択
            retTargetQs = mainQN.model.predict(state, verbose=0)[0]
            action = np.argmax(retTargetQs)
        else:
            # ランダムな行動の選択
            action = np.random.choice([0, 1])
        
        return action

# ==========================================================
# 評価関数
# ==========================================================
def evaluate_without_learning(env, qn, actor_name, evaluation_type, episode_num):
    """
    学習なしでの評価実行
    
    Args:
        env: 評価用の環境
        qn: 評価するQネットワーク
        actor_name: アクター名
        evaluation_type: 評価タイプ（'mid' or 'final'）
        episode_num: 評価時のエピソード番号
    
    Returns:
        EvaluationData: 評価結果
    """
    eval_data = EvaluationData(actor_name, evaluation_type, episode_num)
    state, _ = env.reset()
    state = np.reshape(state, [1, 4])
    done = False
    start_time = time.time()
    
    # 状態履歴の保持
    states_history = []
    
    # エピソードの実行
    while not done and eval_data.steps < MAX_STEPS:
        states_history.append(state[0])
        # 行動の選択
        retTargetQs = qn.model.predict(state, verbose=0)[0]
        action = np.argmax(retTargetQs)
        
        # 行動の記録
        eval_data.action_distribution[action] += 1
        
        # 環境の更新
        next_state, reward, done, _, _ = env.step(action)
        eval_data.total_reward += reward
        
        state = np.reshape(next_state, [1, 4])
        eval_data.steps += 1
    
    # 状態値の統計量を計算
    states_array = np.array(states_history)
    eval_data.average_state_values = np.mean(states_array, axis=0).tolist()
    eval_data.max_state_values = np.max(states_array, axis=0).tolist()
    eval_data.min_state_values = np.min(states_array, axis=0).tolist()
    
    eval_data.time_taken = time.time() - start_time
    return eval_data

# ==========================================================
# メイン処理
# ==========================================================
def main():
    """
    メインの学習・評価ループ
    CartPole環境でマリオとルイージの協調学習を実行
    """
    # ディレクトリ設定
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mario_videos_dir = os.path.join(base_dir, 'mario_videos')
    luigi_videos_dir = os.path.join(base_dir, 'luigi_videos')
    results_dir = os.path.join(base_dir, 'results')
    qvalue_vis_dir = os.path.join(base_dir, 'qvalue_visualizations')  # Q値可視化用ディレクトリ

    # 必要なディレクトリの作成
    for directory in [mario_videos_dir, luigi_videos_dir, results_dir, qvalue_vis_dir]:
        os.makedirs(directory, exist_ok=True)

    # 結果保存用のファイル名を設定（タイムスタンプ付き）
    time_stamp = str(dt.datetime.now().strftime('%Y%m%d%H%M%S'))+str(datetime.datetime.now().microsecond)
    history_fname = os.path.join(results_dir, f'steps_history_{time_stamp}.csv')
    evaluation_fname = os.path.join(results_dir, f'evaluation_results_{time_stamp}.csv')

    # 複数回の試行を実行
    for trial in range(50):
        print(f"\nStarting Trial {trial + 1}/50")
        
        # 環境の初期化
        env = gym.make('CartPole-v1')
        mario_env = gym.wrappers.RecordVideo(
            gym.make('CartPole-v1', render_mode='rgb_array'),
            video_folder=mario_videos_dir,
            name_prefix=f'mario_{trial}_{time_stamp}'
        )
        luigi_env = gym.wrappers.RecordVideo(
            gym.make('CartPole-v1', render_mode='rgb_array'),
            video_folder=luigi_videos_dir,
            name_prefix=f'luigi_{trial}_{time_stamp}'
        )

        # QNetworkの初期化
        mario_mainQN = QNetwork(hidden_size=64, learning_rate=LEARNING_RATE)
        mario_targetQN = QNetwork(hidden_size=64, learning_rate=LEARNING_RATE)
        luigi_mainQN = QNetwork(hidden_size=32, learning_rate=LEARNING_RATE)
        luigi_targetQN = QNetwork(hidden_size=32, learning_rate=LEARNING_RATE)

        # 経験メモリとアクター管理の初期化
        memory = Memory(max_size=MEMORY_SIZE)
        actor = CollaborativeActor()
        
        # データ記録用の配列初期化
        steps_history = []
        actors_history = []
        evaluation_results = []
        
        # 学習の終了判定用変数
        total_reward_vec = np.zeros(10)
        goal_average_reward = 195
        islearned = 0
        
        # エピソードループ
        episode = 0
        while True:
            # 環境のリセット
            state, _ = env.reset()
            state = np.reshape(state, [1, 4])
            episode_start_time = time.time()
            
            # エピソード開始時にアクターを選択
            current_actor = actor.get_actor()
            
            # 中間評価（50エピソード時点）
            if episode == EVALUATION_POINT:
                print("\n50エピソード時点での評価:")
                # マリオの評価
                mario_eval = evaluate_without_learning(mario_env, mario_mainQN, 'mario', 'mid', episode)
                evaluation_results.append(mario_eval)
                print(f"マリオの中間評価結果: {mario_eval.steps}ステップ")
                
                # ルイージの評価
                luigi_eval = evaluate_without_learning(luigi_env, luigi_mainQN, 'luigi', 'mid', episode)
                evaluation_results.append(luigi_eval)
                print(f"ルイージの中間評価結果: {luigi_eval.steps}ステップ")
                
                # 50エピソード時点でQ値を可視化（最初のトライアルのみ）
                if trial == 0:
                    print("50エピソード時点のQ値を可視化中...")
                    visualize_q_values(mario_mainQN, luigi_mainQN, episode, qvalue_vis_dir)
                    compare_q_values(mario_mainQN, luigi_mainQN, episode, qvalue_vis_dir)
                
                episode += 1
                continue
            
            # 最終評価（学習完了時または最大エピソード到達時）
            if islearned or episode >= NUM_EPISODES:
                print("\n最終評価:")
                # マリオの最終評価
                mario_final = evaluate_without_learning(mario_env, mario_mainQN, 'mario', 'final', episode)
                evaluation_results.append(mario_final)
                print(f"マリオの最終評価結果: {mario_final.steps}ステップ")
                
                # ルイージの最終評価
                luigi_final = evaluate_without_learning(luigi_env, luigi_mainQN, 'luigi', 'final', episode)
                evaluation_results.append(luigi_final)
                print(f"ルイージの最終評価結果: {luigi_final.steps}ステップ")
                
                # 100エピソード時点でQ値を可視化（最初のトライアルのみ）
                if trial == 0:
                    print("100エピソード時点のQ値を可視化中...")
                    visualize_q_values(mario_mainQN, luigi_mainQN, episode, qvalue_vis_dir)
                    compare_q_values(mario_mainQN, luigi_mainQN, episode, qvalue_vis_dir)
                
                # 学習履歴の保存
                if steps_history and actors_history:
                    safe_write_csv(history_fname, [steps_history, actors_history])
                
                # 評価結果の保存
                evaluation_headers = get_evaluation_headers()
                
                if trial == 0:
                    safe_write_csv(evaluation_fname, None, evaluation_headers, mode='w')
                
                for eval_data in evaluation_results:
                    formatted_data = format_evaluation_data(eval_data)
                    safe_write_csv(evaluation_fname, formatted_data)
                
                break
            
            # ターゲットネットワークの更新
            mario_targetQN.model.set_weights(mario_mainQN.model.get_weights())
            luigi_targetQN.model.set_weights(luigi_mainQN.model.get_weights())
            
            # エピソード内のループ
            done = False
            steps = 0
            episode_reward = 0
            
            while not done and steps < MAX_STEPS:
                # 現在のアクターで行動を選択と実行
                action = actor.get_action(state, mario_mainQN, luigi_mainQN, current_actor, episode)
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                
                # 報酬の設定
                if done:
                    next_state = np.zeros(state.shape)
                    if steps < 195:
                        reward = -1
                    else:
                        reward = 1
                else:
                    reward = 0
                    
                # 経験の記録と更新
                episode_reward += 1
                memory.add((state, action, reward, next_state))
                state = next_state
                steps += 1
                
                # 経験リプレイによる学習
                if memory.len() > BATCH_SIZE:
                    mario_mainQN.replay(memory, BATCH_SIZE, GAMMA, mario_targetQN)
                    luigi_mainQN.replay(memory, BATCH_SIZE, GAMMA, luigi_targetQN)

            # エピソード結果の記録
            steps_history.append(steps)
            actors_history.append(current_actor)
            actor.update_steps(current_actor, steps)
            
            # 進捗の表示
            episode_time = time.time() - episode_start_time
            print(f"Trial {trial + 1}, Episode {episode}: {current_actor} achieved {steps} steps in {episode_time:.2f} seconds")
            
            # 終了判定用の報酬記録
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))
            
            # 学習完了判定
            if total_reward_vec.mean() >= goal_average_reward:
                print(f'Trial {trial + 1}, Episode {episode} train agent successfully!')
                islearned = 1
            
            episode += 1

        # 環境のクローズ
        env.close()
        mario_env.close_video_recorder()
        mario_env.close()
        luigi_env.close_video_recorder()
        luigi_env.close()

# メイン処理の実行
if __name__ == "__