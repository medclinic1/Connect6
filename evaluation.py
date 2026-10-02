from tools import *

# ---------------------------------------------------------------------------
# Ventanas de 6 casillas (compartidas con move_generator.py: NO renombrar)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Parámetros de la evaluación mejorada (sesión 2) -> candidatos a optimizar (sesión 3)
# ---------------------------------------------------------------------------
THREAT_STONES = 4        # Ventana viva con >= 4 piedras: se completa en un turno (2 piedras)
WIN_SCORE = 15000        # Victoria "anunciada" (aún no jugada): < MAXINT para distinguirla de una real
QUIET_LIMIT = 9000       # Cota de la parte posicional: nunca alcanza a WIN_SCORE
TEMPO = 1.25             # Bonus de iniciativa para el jugador que mueve
FORCED_PENALTY = 200     # Coste por cada piedra que el que mueve debe gastar en defender


# ---------------------------------------------------------------------------
# Evaluación naive (sesión 1) -> se conserva para comparar en la experimentación
# ---------------------------------------------------------------------------
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
    return score


def _terminal_value(board, our_color, preMove):
    """MAXINT / MININT / 0 si la partida ha terminado; None si sigue."""
    result = check_game_over(board, preMove)
    if result == our_color:
        return Defines.MAXINT
    if result == (our_color ^ 3):
        return Defines.MININT
    if result == Defines.DRAW:
        return 0
    return None


def evaluate_naive(board, our_color, preMove):
    """Evaluación de la sesión 1 (ventanas vivas, sin amenazas ni turno)."""
    terminal = _terminal_value(board, our_color, preMove)
    if terminal is not None:
        return terminal
    score = heuristic(board, our_color)
    return max(Defines.MININT + 1, min(Defines.MAXINT - 1, score))


# ---------------------------------------------------------------------------
# Evaluación mejorada (sesión 2)
# ---------------------------------------------------------------------------
def analyze_board(board):
    """Una sola pasada por las 924 ventanas. Para cada color devuelve:
       - quiet[c]  : suma de WEIGHTS de sus ventanas vivas
       - threat[c] : lista de ventanas amenazantes (>= 4 piedras, vivas),
                     cada una como frozenset de sus casillas vacías."""
    B, W = Defines.BLACK, Defines.WHITE
    quiet = {B: 0, W: 0}
    threat = {B: [], W: []}
    for w in WINDOWS:
        vals = [board[x][y] for x, y in w]
        nb = vals.count(B)
        nw = vals.count(W)
        if nb and not nw:
            owner, n = B, nb
        elif nw and not nb:
            owner, n = W, nw
        else:
            continue                   # vacía o muerta
        quiet[owner] += WEIGHTS[min(n, 5)]   # min(): robustez si ya hubiera un 6
        if n >= THREAT_STONES:
            threat[owner].append(frozenset(c for c, v in zip(w, vals) if v == Defines.NOSTONE))
    return quiet, threat


def count_threats(windows):
    """Nº mínimo de piedras que necesita el rival para bloquear todas las ventanas
    amenazantes (problema de cobertura mínima / hitting set).
    Solo importa distinguir 0, 1, 2 y '3 o más' (el rival pone 2 piedras por turno),
    así que se calcula de forma EXACTA y barata:
      1 -> alguna casilla está en todas las ventanas (intersección no vacía)
      2 -> existe una casilla c tal que las ventanas que no contienen c
           tienen otra casilla en común
      3 -> ninguna de las anteriores (imposible de bloquear con 2 piedras)"""
    if not windows:
        return 0
    if frozenset.intersection(*windows):
        return 1
    for c in frozenset.union(*windows):
        rest = [w for w in windows if c not in w]
        if frozenset.intersection(*rest):
            return 2
    return 3


def side_to_move(board, preMove, our_color):
    """El turno es del rival de quien colocó las piedras de preMove.
    Si preMove no apunta a una piedra (p. ej. raíz sin jugada previa), se asume
    que mueve our_color (solo evaluamos cuando nos toca buscar)."""
    last = board[preMove.positions[0].x][preMove.positions[0].y]
    if last in (Defines.BLACK, Defines.WHITE):
        return last ^ 3
    return our_color


def evaluate(board, our_color, preMove):
    """
    Evaluación estática desde el punto de vista de our_color.
      1. Partida terminada: MAXINT (ganamos), MININT (perdemos), 0 (empate).
      2. Amenazas, según quién mueve (S) y su rival (R):
           - S tiene >= 1 amenaza  -> S gana en su turno         (+WIN_SCORE para S)
           - R tiene >= 3 amenazas -> S no puede bloquearlas     (-WIN_SCORE para S)
      3. Posición tranquila: ventanas vivas ponderadas, con bonus de iniciativa
         para S y penalización si S está obligado a defender (R con 1-2 amenazas).
    """
    terminal = _terminal_value(board, our_color, preMove)
    if terminal is not None:
        return terminal

    mover = side_to_move(board, preMove, our_color)
    rival = mover ^ 3
    quiet, threat = analyze_board(board)

    if threat[mover]:                                  # S completa un 6 en su turno
        value = WIN_SCORE
    else:
        rival_threats = count_threats(threat[rival])
        if rival_threats >= 3:                         # S no puede tapar todo
            value = -WIN_SCORE
        else:
            value = TEMPO * quiet[mover] - quiet[rival] - FORCED_PENALTY * rival_threats
            value = max(-QUIET_LIMIT, min(QUIET_LIMIT, value))

    value = int(value)
    return value if mover == our_color else -value
