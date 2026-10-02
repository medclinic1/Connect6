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
        # Number of nodes explored
        self.m_total_nodes += 1

        # Determine which turn is it (even = our turn) (MAX)
        # (odd = rival's turn) (MIN)
        is_max_node = (self.m_alphabeta_depth - depth) % 2 == 0
        current_color = ourColor if is_max_node else (ourColor ^ 3)

        #if game is over or we reached 0 depth (limit depth) stop the search.
        if depth == 0 or check_game_over(self.m_board, preMove) is not None:
            return evaluate(self.m_board, ourColor, preMove) #Otherwise return heuristic value

        #generate cantidades, limited by max_moves and max_cells (naive proposal)
        moves = generate_moves(self.m_board, current_color)

        #Min-Max basic algorithm
        if is_max_node: #our turn
            best_score = Defines.MININT
            for move in moves:
                # Simulate move
                make_move(self.m_board, move, current_color)
                # recursive call exploring +1 depth in tree.
                score = self.alpha_beta_search(depth - 1, alpha, beta, ourColor, bestMove, move)
                # Unmake_move until we are sure it's the right move
                unmake_move(self.m_board, move)

                if score > best_score:
                    best_score = score
                    # Update best move only if we are at max depth defined.
                    if depth == self.m_alphabeta_depth:
                        bestMove.positions[0].x = move.positions[0].x
                        bestMove.positions[0].y = move.positions[0].y
                        bestMove.positions[1].x = move.positions[1].x
                        bestMove.positions[1].y = move.positions[1].y
            return best_score
            
        else:
            # Rival's turn (minimize our score)
            best_score = Defines.MAXINT
            for move in moves:
                make_move(self.m_board, move, current_color)
                score = self.alpha_beta_search(depth - 1, alpha, beta, ourColor, bestMove, move)
                unmake_move(self.m_board, move)

                if score < best_score:
                    best_score = score
            return best_score

def flush_output():
    import sys
    sys.stdout.flush()
