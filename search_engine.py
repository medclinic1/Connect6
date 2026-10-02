from tools import *
from evaluation import evaluate
from move_generator import generate_moves

class SearchEngine():
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_alphabeta_depth = None
        self.m_total_nodes = 0

    def before_search(self, board, color, alphabeta_depth):
        self.m_board = [row[:] for row in board]
        self.m_chess_type = color
        self.m_alphabeta_depth = alphabeta_depth
        self.m_total_nodes = 0

    def alpha_beta_search(self, depth, alpha, beta, ourColor, bestMove, preMove):
        # 1. Contabilizamos el nodo para las métricas de rendimiento
        self.m_total_nodes += 1
        
        # 2. Determinamos de quién es el turno
        is_max_node = (self.m_alphabeta_depth - depth) % 2 == 0
        current_color = ourColor if is_max_node else (ourColor ^ 3)

        # 3. Condición de parada (límite de profundidad o fin del juego)
        if depth == 0 or check_game_over(self.m_board, preMove) is not None:
            return evaluate(self.m_board, ourColor, preMove)

        # 4. Generar jugadas candidatas
        moves = generate_moves(self.m_board, current_color, max_moves=20)

        # --- NODO MAX (Nuestro turno: buscamos maximizar la puntuación) ---
        if is_max_node:
            best_score = Defines.MININT
            for move in moves:
                make_move(self.m_board, move, current_color)
                score = self.alpha_beta_search(depth - 1, alpha, beta, ourColor, bestMove, move)
                unmake_move(self.m_board, move)

                if score > best_score:
                    best_score = score
                    # Solo guardamos el movimiento real si estamos en la raíz del árbol
                    if depth == self.m_alphabeta_depth:
                        bestMove.positions[0].x = move.positions[0].x
                        bestMove.positions[0].y = move.positions[0].y
                        bestMove.positions[1].x = move.positions[1].x
                        bestMove.positions[1].y = move.positions[1].y
                
                # PODA ALFA-BETA: Actualizamos alfa y podamos si es necesario
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break # El oponente nunca nos dejará llegar a esta rama, dejamos de explorar
                    
            return best_score
            
        # --- NODO MIN (Turno del rival: busca minimizar nuestra puntuación) ---
        else:
            best_score = Defines.MAXINT
            for move in moves:
                make_move(self.m_board, move, current_color)
                score = self.alpha_beta_search(depth - 1, alpha, beta, ourColor, bestMove, move)
                unmake_move(self.m_board, move)

                if score < best_score:
                    best_score = score
                
                # PODA ALFA-BETA: Actualizamos beta y podamos si es necesario
                beta = min(beta, best_score)
                if beta <= alpha:
                    break # Nosotros nunca elegiremos llegar a esta rama, dejamos de explorar
                    
            return best_score

def flush_output():
    import sys
    sys.stdout.flush()
