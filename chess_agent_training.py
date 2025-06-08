import time
import pandas as pd
import matplotlib.pyplot as plt
import chess
import chess.engine
import random
import os

from chess_model import ChessAgent

def Q_learning(agent, stockfish_path, games_to_play, max_game_moves, board_config=None):
    if not os.path.exists(stockfish_path):
        print(f"Stockfish motoru bulunamadı: {stockfish_path}")
        print("Lütfen doğru yolu belirtin.")
        return

    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    loss = []
    final_score = []
    games = 0
    steps = 0
    start_time = time.time()

    # we play n games
    while games < games_to_play:
        games += 1
        print(f"Oyun {games}/{games_to_play} başlıyor...")

        # Create a new standard board
        if board_config is None:
            board = chess.Board()
        else:
            board = chess.Board(board_config)

        done = False
        game_moves = 0

        # analyse board with stockfish
        try:
            analysis = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))
        except chess.engine.EngineTerminatedError:
            print("Stockfish motoru sonlandırıldı. Yeniden başlatılıyor...")
            stockfish.quit()
            stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            analysis = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))


        # get best possible move according to stockfish (with depth=5)
        best_move = analysis['pv'][0]

        # until game is not finished
        while not done:
            game_moves += 1
            steps += 1

            # choose action, here the agent choose whether to explore or exploit
            action_index, move, bit_state, valid_move_tensor = agent.select_action(board, best_move)

            # save this score to compute the reward after the opponent move
            board_score_before = analysis['score'].relative.score(mate_score=10000) / 100

            # white moves
            board.push(move)

            # the game is finished (checkmate, stalemate, draw conditions, ...) or we reached max moves
            done = board.result() != '*' or game_moves > max_game_moves
            
            if done:
                final_result = board.result()
                
                if final_result == '*' or final_result == "1/2-1/2":
                    reward = -10
                elif final_result == "1-0":
                    reward = 1000
                else:
                    reward = -1000

                agent.remember(agent.MAX_PRIORITY, bit_state, action_index, reward, None, done, valid_move_tensor, None)
                board_score_after = reward
            else:
                # black moves
                board.push(random.choice(list(board.legal_moves)))
                
                try:
                    analysis = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))
                except chess.engine.EngineTerminatedError:
                    print("Stockfish motoru sonlandırıldı. Yeniden başlatılıyor...")
                    stockfish.quit()
                    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                    analysis = stockfish.analyse(board=board, limit=chess.engine.Limit(depth=5))

                board_score_after = analysis['score'].relative.score(mate_score=10000) / 100
                done = board.result() != '*'
                
                if not done:
                    best_move = analysis['pv'][0]

                next_bit_state = agent.convert_state(board)
                next_valid_move_tensor, _ = agent.mask_and_valid_moves(board)
                
                reward = board_score_after - board_score_before - 0.01
                                
                agent.remember(agent.MAX_PRIORITY, bit_state, action_index, reward, next_bit_state, done, valid_move_tensor, next_valid_move_tensor)

            loss_val = agent.learn_experience_replay(debug=False)
            if loss_val:
                loss.append(loss_val)

            agent.adaptiveEGreedy()

        final_score.append(board_score_after)
        print(f"Oyun {games} bitti. Skor: {board_score_after:.2f}, Epsilon: {agent.epsilon:.2f}")


    stockfish.quit()
    
    if not loss or not final_score:
        print("Eğitim sırasında yeterli veri toplanmadı, grafikler oluşturulamıyor.")
        return

    # plot training results
    score_df = pd.DataFrame(final_score, columns=["score"])
    score_df['ma'] = score_df["score"].rolling(window = max(1, games_to_play // 5)).mean()
    loss_df = pd.DataFrame(loss, columns=["loss"])
    loss_df['ma'] = loss_df["loss"].rolling(window=max(1, steps // 5)).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(score_df.index, score_df["score"], linewidth=0.2)
    ax1.plot(score_df.index, score_df["ma"])
    ax1.set_title('Oyun sonu skoru')
    ax1.set_xlabel('Oyun')
    ax1.set_ylabel('Skor')

    ax2.plot(loss_df.index, loss_df["loss"], linewidth=0.1)
    ax2.plot(loss_df.index, loss_df["ma"])
    ax2.set_title('Adım başına kayıp (Loss)')
    ax2.set_xlabel('Adım')
    ax2.set_ylabel('Kayıp')

    plt.show()

def train():
    agent = ChessAgent()
    
    # Kullanıcının sistemine göre stockfish yolunu ayarlayın
    stockfish_path = "/Users/zaferavci/Documents/GitHub/Deep Learning/Kaggle/stockfish/stockfish-macos-m1-apple-silicon"
    
    print("Eğitim başlıyor...")
    Q_learning(agent, stockfish_path, games_to_play=50, max_game_moves=75)
    
    model_path = "chess_agent_50_games.pth"
    agent.save_model(model_path)
    print(f"Model şuraya kaydedildi: {model_path}")

if __name__ == "__main__":
    train() 