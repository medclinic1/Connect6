from defines import *
from tools import *
import sys
from search_engine import SearchEngine
import time

# Control de tiempo (sesión 4). El comando "depth d" ya no fija la profundidad:
# fija el tiempo por turno (enunciado, sección 5), y la profundidad la decide la
# profundización iterativa según lo que dé tiempo a calcular.
SECONDS_PER_DEPTH_UNIT = 1.0   # "depth d" -> d segundos por turno (la GUI manda 4, 5 o 6)
DEFAULT_TIME_LIMIT = 5.0       # Si la GUI no manda "depth"
MAX_TIME_LIMIT = 25.0          # La GUI da timeout a los 30 s: nunca pedimos más de 25
SAFETY_MARGIN = 0.3            # Margen para la última llamada y la comunicación con la GUI

class GameEngine:
    def __init__(self, name=Defines.ENGINE_NAME):
        if name and len(name) > 0:
            if len(name) < Defines.MSG_LENGTH:
                self.m_engine_name = name
            else:
                print(f"Too long Engine Name: {name}, should be less than: {Defines.MSG_LENGTH}")
        self.m_time_limit = DEFAULT_TIME_LIMIT
        self.m_board = t = [ [0]*Defines.GRID_NUM for i in range(Defines.GRID_NUM)]
        self.init_game()
        self.m_search_engine = SearchEngine()
        self.m_best_move = StoneMove()

    def init_game(self):
        init_board(self.m_board)

    def on_help(self):
        print(
            f"On help for GameEngine {self.m_engine_name}\n"
            " name        - print the name of the Game Engine.\n"
            " print       - print the board.\n"
            " exit/quit   - quit the game.\n"
            " black XXXX  - place the black stone on the position XXXX on the board.\n"
            " white XXXX  - place the white stone on the position XXXX on the board, X is from A to S.\n"
            " next        - the engine will search the move for the next step.\n"
            " move XXXX   - tell the engine that the opponent made the move XXXX,\n"
            "              and the engine will search the move for the next step.\n"
            " new black   - start a new game and set the engine to black player.\n"
            " new white   - start a new game and set it to white.\n"
            " depth d     - set the time limit per move to d seconds (default 5).\n"
            " vcf         - set vcf search.\n"
            " unvcf       - set none vcf search.\n"
            " help        - print this help.\n")

    def run(self):
        msg = ""
        self.on_help()
        while True:
            try:
                msg = input().strip()
            except EOFError:
                # La GUI cierra la entrada estándar al liberar el motor
                # (al empezar otra partida o al cerrar): salimos sin error.
                break
            log_to_file(msg)
            if msg == "name":
                print(f"name {self.m_engine_name}")
            elif msg == "exit" or msg == "quit":
                break
            elif msg == "print":
                print_board(self.m_board, self.m_best_move)
            elif msg == "vcf":
                self.m_vcf = True
            elif msg == "unvcf":
                self.m_vcf = False
            elif msg.startswith("black"):
                self.m_best_move = msg2move(msg[6:])
                make_move(self.m_board, self.m_best_move, Defines.BLACK)
                self.m_chess_type = Defines.BLACK
            elif msg.startswith("white"):
                self.m_best_move = msg2move(msg[6:])
                make_move(self.m_board, self.m_best_move, Defines.WHITE)
                self.m_chess_type = Defines.WHITE
            elif msg == "next":
                self.m_chess_type = self.m_chess_type ^ 3
                if self.search_a_move(self.m_chess_type, self.m_best_move):
                    make_move(self.m_board, self.m_best_move, self.m_chess_type)
                    msg = f"move {move2msg(self.m_best_move)}"
                    print(msg)
                    flush_output()
            elif msg.startswith("new"):
                self.init_game()
                if msg[4:] == "black":
                    self.m_best_move = msg2move("JJ")
                    make_move(self.m_board, self.m_best_move, Defines.BLACK)
                    self.m_chess_type = Defines.BLACK
                    msg = "move JJ"
                    print(msg)
                    flush_output()
                else:
                    self.m_chess_type = Defines.WHITE
            elif msg.startswith("move"):
                self.m_best_move = msg2move(msg[5:])
                make_move(self.m_board, self.m_best_move, self.m_chess_type ^ 3)
                if is_win_by_premove(self.m_board, self.m_best_move):
                    print("We lost!")
                if self.search_a_move(self.m_chess_type, self.m_best_move):
                    msg = f"move {move2msg(self.m_best_move)}"
                    make_move(self.m_board, self.m_best_move, self.m_chess_type)
                    print(msg)
                    flush_output()
            elif msg.startswith("depth"):
                d = float(msg[6:])
                if d > 0:
                    self.m_time_limit = min(d * SECONDS_PER_DEPTH_UNIT, MAX_TIME_LIMIT)
                print(f"Set the time limit to {self.m_time_limit:.1f} s.\n")
            elif msg == "help":
                self.on_help()
        return 0

    def search_a_move(self, ourColor, bestMove):
        score = 0
        start = 0
        end = 0

        start = time.perf_counter()
        budget = max(0.1, self.m_time_limit - SAFETY_MARGIN)
        score = self.m_search_engine.iterative_deepening(self.m_board, ourColor, bestMove, bestMove, budget)
        end = time.perf_counter()

        print(f"AB Time:\t{end - start:.3f}")
        print(f"Depth:\t{self.m_search_engine.m_depth_reached}")
        if self.m_search_engine.m_tss_info:
            print(f"TSS:\t{self.m_search_engine.m_tss_info}")
        print(f"Node:\t{self.m_search_engine.m_total_nodes}")
        print(f"Pruned:\t{self.m_search_engine.m_pruned}")
        print(f"Cutoffs:\t{self.m_search_engine.m_cutoffs}\n")
        print(f"Score:\t{score:.3f}")
        return True

def flush_output():
    sys.stdout.flush()

# Create an instance of GameEngine and run the game
if __name__ == "__main__":
    game_engine = GameEngine()
    game_engine.run()
