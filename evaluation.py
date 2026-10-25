from tools import *

# ---------------------------------------------------------------------------
# Ventanas de 6 casillas (compartidas con move_generator.py: NO renombrar)
# ---------------------------------------------------------------------------
WINDOW = 6                          # Se gana con 6 en raya -> ventanas de 6 casillas
WEIGHTS = [0, 1, 4, 16, 64, 256]    # Pesos de la evaluación naive (sesión 1)
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
# Constantes tácticas (NO se optimizan: definen la escala de la evaluación)
# ---------------------------------------------------------------------------
THREAT_STONES = 4        # Ventana viva con >= 4 piedras: se completa en un turno (2 piedras)
WIN_SCORE = 15000        # Victoria "anunciada" (aún no jugada): < MAXINT para distinguirla de una real
QUIET_LIMIT = 9000       # Cota de la parte posicional: nunca alcanza a WIN_SCORE

# ---------------------------------------------------------------------------
# Evaluación ponderada (sesión 3): vector de pesos -> optimizable (sesión 4, AG)
# ---------------------------------------------------------------------------
# Cada peso multiplica una característica de la posición. Los valores por
# defecto reproducen EXACTAMENTE la evaluación de la sesión 2 (CONNECT = 0,
# DEFENSE = 1), que sirve de punto de partida ("configuración conocida") para
# inicializar la población del algoritmo genético.
WEIGHT_NAMES = [
    "W1", "W2", "W3", "W4", "W5",   # valor de una ventana viva con 1..5 piedras
    "CONNECT",   # valor de cada pareja de piedras propias contiguas en una ventana viva
    "TEMPO",     # multiplicador de iniciativa para el jugador que mueve
    "DEFENSE",   # multiplicador del potencial del rival (>1 = más defensivo)
    "FORCED",    # coste por cada piedra que el que mueve debe gastar en defender
]
DEFAULT_WEIGHTS = [1, 4, 16, 64, 256, 0, 1.25, 1.0, 200]

_weights = list(DEFAULT_WEIGHTS)

def set_weights(vector):
    """Cambia el vector de pesos activo (lo usará el algoritmo genético)."""
    assert len(vector) == len(WEIGHT_NAMES)
    _weights[:] = vector

def get_weights():
    return list(_weights)

# Compatibilidad: nombres de la sesión 2
TEMPO = DEFAULT_WEIGHTS[WEIGHT_NAMES.index("TEMPO")]
FORCED_PENALTY = DEFAULT_WEIGHTS[WEIGHT_NAMES.index("FORCED")]


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
# Extracción de características
# ---------------------------------------------------------------------------
def extract_features(board):
    """Una sola pasada por las 924 ventanas. Para cada color c devuelve:
       - live[c][n]  : nº de ventanas vivas de c con n piedras (n = 1..5)
       - connect[c]  : nº de parejas de piedras de c contiguas dentro de sus
                       ventanas vivas (conectividad, idea de Liu: piedras
                       cercanas = menos huecos para que el rival corte)
       - threat[c]   : ventanas amenazantes (>= 4 piedras, vivas), cada una
                       como frozenset de sus casillas vacías"""
    B, W = Defines.BLACK, Defines.WHITE
    live = {B: [0] * 6, W: [0] * 6}
    connect = {B: 0, W: 0}
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
        live[owner][min(n, 5)] += 1    # min(): robustez si ya hubiera un 6
        if n >= 2:                     # con 1 piedra no puede haber parejas
            connect[owner] += sum(1 for k in range(WINDOW - 1)
                                  if vals[k] == owner and vals[k + 1] == owner)
        if n >= THREAT_STONES:
            threat[owner].append(frozenset(c for c, v in zip(w, vals) if v == Defines.NOSTONE))
    return live, connect, threat


def analyze_board(board):
    """Compatibilidad con la sesión 2: (quiet, threat) con los pesos activos."""
    live, connect, threat = extract_features(board)
    return {c: potential(live[c], connect[c], _weights) for c in live}, threat


def potential(live_c, connect_c, wv):
    """Potencial posicional de un color: Σ W_n·(ventanas vivas con n) + CONNECT·conectividad."""
    return (wv[0] * live_c[1] + wv[1] * live_c[2] + wv[2] * live_c[3]
            + wv[3] * live_c[4] + wv[4] * live_c[5] + wv[5] * connect_c)


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


# ---------------------------------------------------------------------------
# Evaluación ponderada (sesión 3)
# ---------------------------------------------------------------------------
def evaluate(board, our_color, preMove, weights=None, features=None):
    """
    features: un IncrementalFeatures ya actualizado con 'board' (lo usa la
    búsqueda para no recorrer las 924 ventanas en cada hoja). Si es None, las
    características se calculan desde cero con extract_features().

    Evaluación estática desde el punto de vista de our_color.
      1. Partida terminada: MAXINT (ganamos), MININT (perdemos), 0 (empate).
      2. Reglas tácticas (no ponderadas), según quién mueve (S) y su rival (R):
           - S tiene >= 1 amenaza  -> S gana en su turno         (+WIN_SCORE para S)
           - R tiene >= 3 amenazas -> S no puede bloquearlas     (-WIN_SCORE para S)
      3. Posición tranquila: suma ponderada de características
           TEMPO·P(S) − DEFENSE·P(R) − FORCED·amenazas(R)
         con P(c) = Σ W_n·vivas_n(c) + CONNECT·conectividad(c), acotada a ±QUIET_LIMIT.
    """
    wv = _weights if weights is None else weights
    terminal = _terminal_value(board, our_color, preMove)
    if terminal is not None:
        return terminal

    mover = side_to_move(board, preMove, our_color)
    rival = mover ^ 3
    if features is None:
        live, connect, threat = extract_features(board)
    else:
        live, connect, threat = features.snapshot(board)

    if threat[mover]:                                  # S completa un 6 en su turno
        value = WIN_SCORE
    else:
        rival_threats = count_threats(threat[rival])
        if rival_threats >= 3:                         # S no puede tapar todo
            value = -WIN_SCORE
        else:
            value = (wv[6] * potential(live[mover], connect[mover], wv)
                     - wv[7] * potential(live[rival], connect[rival], wv)
                     - wv[8] * rival_threats)
            value = max(-QUIET_LIMIT, min(QUIET_LIMIT, value))

    value = int(value)
    return value if mover == our_color else -value


# ---------------------------------------------------------------------------
# Características incrementales (sesión 4: optimización de la velocidad)
# ---------------------------------------------------------------------------
# Al poner una piedra solo cambian las ventanas que contienen esa casilla
# (como mucho 24 de las 924). En vez de recorrer el tablero entero en cada hoja,
# se mantienen los totales (vivas por nº de piedras, conectividad, ventanas
# amenazantes) y se corrigen solo con las ventanas afectadas en cada make/unmake.
CELL_WINDOWS = {}
for _idx, _w in enumerate(WINDOWS):
    for _c in _w:
        CELL_WINDOWS.setdefault(_c, []).append(_idx)


def _window_contrib(board, w):
    """(dueño, nº de piedras, parejas contiguas) de una ventana viva; None si está
    vacía o muerta. Es exactamente lo que suma extract_features() por ventana."""
    vals = [board[x][y] for x, y in w]
    nb = vals.count(Defines.BLACK)
    nw = vals.count(Defines.WHITE)
    if nb and not nw:
        owner, n = Defines.BLACK, nb
    elif nw and not nb:
        owner, n = Defines.WHITE, nw
    else:
        return None
    conn = 0
    if n >= 2:
        conn = sum(1 for k in range(WINDOW - 1) if vals[k] == owner and vals[k + 1] == owner)
    return owner, n, conn


class IncrementalFeatures:
    def __init__(self, board):
        B, W = Defines.BLACK, Defines.WHITE
        self.live = {B: [0] * 6, W: [0] * 6}
        self.connect = {B: 0, W: 0}
        self.threat_idx = {B: set(), W: set()}     # índices de ventanas con >= 4 piedras
        for idx, w in enumerate(WINDOWS):
            self._apply(idx, _window_contrib(board, w), +1)

    def _apply(self, idx, contrib, sign):
        if contrib is None:
            return
        owner, n, conn = contrib
        self.live[owner][min(n, 5)] += sign
        self.connect[owner] += sign * conn
        if n >= THREAT_STONES:
            if sign > 0:
                self.threat_idx[owner].add(idx)
            else:
                self.threat_idx[owner].discard(idx)

    def set_cell(self, board, x, y, color):
        """board[x][y] = color, manteniendo los totales al día (color 0 = quitar)."""
        affected = CELL_WINDOWS[(x, y)]
        for idx in affected:
            self._apply(idx, _window_contrib(board, WINDOWS[idx]), -1)
        board[x][y] = color
        for idx in affected:
            self._apply(idx, _window_contrib(board, WINDOWS[idx]), +1)

    def make(self, board, move, color):
        p, q = move.positions
        self.set_cell(board, p.x, p.y, color)
        if (p.x, p.y) != (q.x, q.y):
            self.set_cell(board, q.x, q.y, color)

    def unmake(self, board, move):
        p, q = move.positions
        self.set_cell(board, p.x, p.y, Defines.NOSTONE)
        if (p.x, p.y) != (q.x, q.y):
            self.set_cell(board, q.x, q.y, Defines.NOSTONE)

    def snapshot(self, board):
        """Lo mismo que devuelve extract_features(board), sin recorrer el tablero."""
        threat = {}
        for c, idxs in self.threat_idx.items():
            threat[c] = [frozenset(cell for cell in WINDOWS[i] if board[cell[0]][cell[1]] == Defines.NOSTONE)
                         for i in idxs]
        return self.live, self.connect, threat
