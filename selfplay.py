"""
Partidas dentro del mismo proceso entre dos vectores de pesos de la evaluación
(sesión 4). Es la pieza que usa el algoritmo genético para calcular el fitness:
mucho más rápida que lanzar dos motores por la interfaz de texto (arena.py).

La búsqueda es a profundidad FIJA (no por tiempo): así una partida es
determinista y reproducible, no depende de la carga de la máquina, y la única
diferencia entre los dos jugadores son sus pesos.
"""
from defines import Defines, StoneMove
from tools import init_board, make_move, msg2move, check_game_over
import evaluation
from search_engine import SearchEngine

MAX_TURNS = 120


def play_weights(w_black, w_white, opening, depth=2, max_turns=MAX_TURNS, time_limit=None):
    """
    Juega una partida entre dos vectores de pesos desde 'opening'
    (lista de (color, "XXXX"), como en arena.make_openings).
    Devuelve (ganador, nº de jugadas): ganador = BLACK / WHITE / DRAW.
    time_limit: si se da, cada jugada usa la profundización iterativa con ese
    tiempo (como el motor real) en vez de la profundidad fija 'depth'.
    """
    board = [[0] * Defines.GRID_NUM for _ in range(Defines.GRID_NUM)]
    init_board(board)
    last, pre = None, None
    for color, msg in opening:
        pre = msg2move(msg)
        make_move(board, pre, color)
        last = color
    mover = last ^ 3
    weights = {Defines.BLACK: w_black, Defines.WHITE: w_white}
    se = SearchEngine()
    turns = len(opening)
    previous = evaluation.get_weights()
    try:
        while True:
            evaluation.set_weights(weights[mover])   # cada jugador evalúa con SUS pesos
            best = StoneMove()
            if time_limit:
                se.iterative_deepening(board, mover, best, pre, time_limit)
            else:
                se.before_search(board, mover, depth)
                se.alpha_beta_search(depth, Defines.MININT, Defines.MAXINT, mover, best, pre)
            make_move(board, best, mover)
            pre = best
            turns += 1
            result = check_game_over(board, best)
            if result is not None:
                return result, turns
            if turns >= max_turns:
                return Defines.DRAW, turns
            mover ^= 3
    finally:
        evaluation.set_weights(previous)
