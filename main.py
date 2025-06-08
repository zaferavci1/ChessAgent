import torch
import chess
import random
import os
from chess_model import ChessAgent

def choose_move(agent, board):
    """
    Ajanın mevcut pozisyon için en iyi hamleyi seçmesini sağlar (keşif olmadan).
    """
    if not isinstance(agent, ChessAgent):
        raise TypeError("Ajan, ChessAgent sınıfının bir örneği olmalıdır.")

    bit_state = agent.convert_state(board)
    valid_moves_tensor, valid_move_dict = agent.mask_and_valid_moves(board)

    with torch.no_grad():
        tensor = torch.from_numpy(bit_state).float().unsqueeze(0)
        policy_values = agent.policy_net(tensor, valid_moves_tensor)
        chosen_move_index = int(policy_values.max(1)[1].view(1, 1))
        
        if chosen_move_index in valid_move_dict:
            chosen_move = valid_move_dict[chosen_move_index]
        else:
            chosen_move = random.choice(list(board.legal_moves))
            
    return chosen_move

def solve_puzzle(agent, fen, solution_moves):
    """
    Verilen bir mat problemini çözer ve sonucu yazdırır.
    """
    board = chess.Board(fen)
    print("="*30)
    print("Problem:")
    print(board.unicode())
    print(f"FEN: {board.fen()}")
    
    if board.is_checkmate():
        print("Pozisyon zaten mat.")
        return

    if board.is_stalemate():
        print("Pozisyon zaten pat.")
        return

    agent_move = choose_move(agent, board)
    
    print(f"\nAjanın Hamlesi: {agent_move.uci()}")
    
    is_correct = False
    for sol_move in solution_moves:
        if agent_move.uci() == sol_move:
            is_correct = True
            break
            
    if is_correct:
        print("Sonuç: DOĞRU! Ajan doğru hamleyi buldu.")
    else:
        print(f"Sonuç: YANLIŞ. Beklenen hamlelerden biri: {', '.join(solution_moves)}")
    print("="*30)

def main():
    model_path = "chess_agent_50_games.pth"
    if not os.path.exists(model_path):
        print(f"Model dosyası bulunamadı: {model_path}")
        print("Lütfen önce 'python chess_agent_training.py' komutunu çalıştırarak modeli eğitin.")
        return

    try:
        agent = ChessAgent(input_model_path=model_path)
        agent.policy_net.eval()  # Modeli değerlendirme moduna al
        print(f"'{model_path}' adresinden model başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return
        
    # Mate in 1, 2 and 3 puzzles
    puzzles = [
        {
            "name": "Mate in 1 (Simple Rook)",
            "fen": "k7/8/1K6/8/8/4R3/8/8 w - - 0 1",
            "solutions": ["e3e8"]
        },
        {
            "name": "Mate in 1 (Simple Queen)",
            "fen": "k7/8/K7/8/8/8/Q7/8 w - - 0 1",
            "solutions": ["a2a8"]
        },
        {
            "name": "Mate in 1 (Two Rooks)",
            "fen": "k7/R7/K7/8/R7/8/8/8 w - - 0 1",
            "solutions": ["a4a5"]
        },
        {
            "name": "Mate in 2 (Puzzle 1)",
            "fen": "1r2k1r1/4bp2/p1p1p3/q1p1P3/2P2P1p/P4Q1P/1P1R2P1/3R2K1 w - - 1 29",
            "solutions": ["f3c6"] 
        },
        {
            "name": "Mate in 2 (Puzzle 2)",
            "fen": "r1b1k1r1/p3pp1p/2p5/q1p1b3/3p3P/P2P2P1/1PP1NP2/R1BQK2R w KQq - 1 14",
            "solutions": ["d1d2"]
        },
        {
            "name": "Mate in 3 (Puzzle 1)",
            "fen": "r3k2r/p6p/P1p2qp1/2p1p3/3n4/2QP4/1P3PPP/R3R1K1 w kq - 0 21",
            "solutions": ["e1e5"]
        },
        {
            "name": "Mate in 3 (Puzzle 2)",
            "fen": "8/8/4k1p1/p1p1P2p/P1P2P1P/3K4/8/8 w - - 0 50",
            "solutions": ["d3e4"]
        }
    ]
    
    for puzzle in puzzles:
        print(f"\n--- {puzzle['name']} ---")
        solve_puzzle(agent, puzzle["fen"], puzzle["solutions"])

if __name__ == "__main__":
    main() 