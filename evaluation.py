from tools import *

WINDOW = 6                          # Se gana con 6 en raya -> ventanas de 6 casillas
WEIGHTS = [0, 1, 4, 16, 64, 256]    # Valor de una ventana viva según nº de piedras (0..5)
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]   # ─  │  ╲  ╱

def _build_windows():
    """Precalcula (una sola vez) las coordenadas de todas las ventanas de 6
    casillas que caben en el tablero, en las 4 direcciones."""
    windows = []
    for x in range(1, Defines.GRID_NUM - 1):
        for y in range(1, Defines.GRID_NUM - 1):
            for dx, dy in DIRECTIONS:
                ex, ey = x + dx * (WINDOW - 1), y + dy * (WINDOW - 1)   # última casilla
                if isValidPos(ex, ey):
                    windows.append([(x + dx * k, y + dy * k) for k in range(WINDOW)])
    return windows

WINDOWS = _build_windows()          # 924 ventanas; se calcula al importar el módulo


def heuristic(board, color):
    """Ventanas vivas de 'color' menos ventanas vivas del rival, ponderadas por WEIGHTS."""
    score = 0
    rival = color ^ 3
    for w in WINDOWS:
        vals = [board[x][y] for x, y in w]
        mine = vals.count(color)
        theirs = vals.count(rival)
        if mine and not theirs:        # viva para nosotros
            score += WEIGHTS[mine]
        elif theirs and not mine:      # viva para el rival
            score -= WEIGHTS[theirs]
        # vacía o con piedras de ambos colores -> no aporta
    return score


def evaluate(board, our_color, preMove):
    """
    Evaluación estática desde el punto de vista de our_color.
      - Partida terminada: MAXINT (ganamos), MININT (perdemos), 0 (empate).
      - Si no: heurística de ventanas vivas, acotada para no confundirse nunca
        con una victoria o derrota real.
    """
    result = check_game_over(board, preMove)
    if result == our_color:
        return Defines.MAXINT
    if result == (our_color ^ 3):
        return Defines.MININT
    if result == Defines.DRAW:
        return 0
    score = heuristic(board, our_color)
    return max(Defines.MININT + 1, min(Defines.MAXINT - 1, score))