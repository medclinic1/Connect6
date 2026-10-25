import time
from tools import *
from evaluation import evaluate, IncrementalFeatures
from move_generator import generate_moves, FORCING_THREATS
from evaluation import WIN_SCORE

WIN_SCORE_QS = WIN_SCORE                       # |valor| >= esto: táctica ya decidida
FORCING_PRIORITY = FORCING_THREATS * 10**7     # puntuación mínima del generador para una jugada que crea >= 2 amenazas

# ---------------------------------------------------------------------------
# Parámetros de la búsqueda con límite de tiempo (sesión 4)
# ---------------------------------------------------------------------------
MAX_ID_DEPTH = 10      # Tope de la profundización iterativa (en la práctica nunca se alcanza)
MAX_MOVES = 20         # Candidatas por nodo que se piden al generador
DEFAULT_EBF = 6.0      # Factor de ramificación efectivo supuesto si aún no se ha medido
USE_TT = True          # Tabla de transposición (se puede desactivar para comparar)
USE_INCREMENTAL = True # Características de la evaluación actualizadas en cada make/unmake
USE_ROOT_ORDER = False # Raíz ordenada por los valores de la iteración anterior (medido: no aporta)
USE_KILLERS = True     # Jugadas asesinas: 2 por nivel, las que provocaron cortes
USE_HISTORY = True     # Heurística histórica: puntos a las jugadas que cortan (2^prof.)
USE_QS = True          # Búsqueda de quiescencia sobre amenazas en las hojas
QS_MAX_PLY = 2         # Jugadas extra como máximo en la quiescencia
QS_MAX_MOVES = 6       # N-best: candidatas por nodo de quiescencia (forward pruning)

# ---------------------------------------------------------------------------
# Hashing de Zobrist (tema 3.3, diap. 20-22)
# ---------------------------------------------------------------------------
# Un número aleatorio de 64 bits por cada (casilla, color). El hash de una
# posición es el XOR de los números de todas sus piedras. Al poner o quitar una
# piedra basta con hacer XOR con su número (actualización incremental, O(1)).
import random as _random
_rng = _random.Random(20261024)            # semilla fija: hashes reproducibles
ZOBRIST = [[[_rng.getrandbits(64) for _ in range(3)]
             for _ in range(Defines.GRID_NUM)] for _ in range(Defines.GRID_NUM)]
ZOBRIST_SIDE = [0, _rng.getrandbits(64), _rng.getrandbits(64)]   # quién mueve

# Tipo de valor guardado en la tabla (alfa-beta no siempre calcula el valor exacto)
EXACT, LOWER, UPPER = 0, 1, 2


def board_hash(board):
    """Hash de Zobrist de un tablero completo (solo al empezar la búsqueda)."""
    h = 0
    for i in range(1, Defines.GRID_NUM - 1):
        for j in range(1, Defines.GRID_NUM - 1):
            c = board[i][j]
            if c == Defines.BLACK or c == Defines.WHITE:
                h ^= ZOBRIST[i][j][c]
    return h


def move_key(move, color):
    """XOR que añade (o quita) las piedras de 'move' al hash. Si las dos piedras
    son la misma casilla (primera jugada de negras), se cuenta una sola vez."""
    p, q = move.positions
    k = ZOBRIST[p.x][p.y][color]
    if (p.x, p.y) != (q.x, q.y):
        k ^= ZOBRIST[q.x][q.y][color]
    return k


def move_cells(move):
    return frozenset(((move.positions[0].x, move.positions[0].y),
                      (move.positions[1].x, move.positions[1].y)))


class SearchTimeout(Exception):
    """Se lanza dentro del árbol cuando se agota el tiempo: aborta la iteración en curso."""
    pass


class SearchEngine():
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_alphabeta_depth = None
        self.m_total_nodes = 0
        self.m_deadline = None         # instante (perf_counter) en que hay que parar; None = sin límite
        self.m_pv_move = None          # mejor jugada de la iteración anterior (se explora primero)
        self.m_depth_reached = 0       # última profundidad COMPLETADA
        self.m_iterations = []         # métricas por iteración: (prof., nodos, tiempo, valor)
        self.m_tt = {}                 # tabla de transposición: hash -> (prof., valor, tipo, jugada)
        self.m_hash = 0                # hash de Zobrist de m_board (se actualiza en cada make/unmake)
        self.m_tt_hits = 0             # nodos resueltos directamente con la tabla
        self.m_use_tt = USE_TT
        self.m_use_incremental = USE_INCREMENTAL
        self.m_feat = None
        self.m_use_root_order = USE_ROOT_ORDER
        self.m_use_killers = USE_KILLERS
        self.m_use_history = USE_HISTORY
        self.m_killers = {}            # nivel (ply) -> [jugada, jugada] (como frozenset de casillas)
        self.m_history = {}            # (color, jugada) -> puntuación acumulada
        self.m_root_scores = {}        # jugada -> valor en la raíz en la iteración anterior
        self.m_root_scores_cur = {}
        self.m_use_qs = USE_QS
        self.m_qs_nodes = 0

    def before_search(self, board, color, alphabeta_depth, deadline=None):
        self.m_board = [row[:] for row in board]
        self.m_chess_type = color
        self.m_alphabeta_depth = alphabeta_depth
        self.m_total_nodes = 0
        self.m_deadline = deadline
        self.m_pv_move = None
        self.m_tt = {}                 # tabla nueva en cada jugada (el valor depende de ourColor)
        self.m_hash = board_hash(self.m_board)
        self.m_tt_hits = 0
        self.m_feat = IncrementalFeatures(self.m_board) if self.m_use_incremental else None
        self.m_killers = {}
        self.m_history = {}
        self.m_root_scores = {}
        self.m_root_scores_cur = {}
        self.m_qs_nodes = 0

    @staticmethod
    def _copy_move(src, dst):
        for k in range(2):
            dst.positions[k].x = src.positions[k].x
            dst.positions[k].y = src.positions[k].y

    @staticmethod
    def _same_move(a, b):
        """Misma jugada aunque las dos piedras vengan en distinto orden."""
        pa = {(p.x, p.y) for p in a.positions}
        pb = {(p.x, p.y) for p in b.positions}
        return pa == pb

    def _make(self, move, color):
        if self.m_feat is not None:
            self.m_feat.make(self.m_board, move, color)    # pone las piedras y actualiza totales
        else:
            make_move(self.m_board, move, color)

    def _unmake(self, move):
        if self.m_feat is not None:
            self.m_feat.unmake(self.m_board, move)
        else:
            unmake_move(self.m_board, move)

    # -----------------------------------------------------------------------
    # Búsqueda de quiescencia sobre amenazas (tema 3.3, diap. 3-5)
    # -----------------------------------------------------------------------
    def _needs_quiescence(self, mover):
        """Una hoja no es tranquila si el rival del que mueve tiene amenazas
        (ventanas vivas con >= 4 piedras): la evaluación estática solo puede
        suponer que se defenderá, pero no sabe qué pasa después de la defensa.
        (Si el que mueve tiene amenaza, gana: la evaluación ya es exacta.)"""
        if not self.m_use_qs or self.m_feat is None:
            return False
        return bool(self.m_feat.threat_idx[mover ^ 3]) and not self.m_feat.threat_idx[mover]

    def _quiescence(self, alpha, beta, ourColor, mover, preMove, qply):
        """
        Extiende la búsqueda solo con jugadas FORZADAS hasta una posición tranquila:
          - Si el que mueve está amenazado: todas sus defensas (no puede "plantarse").
          - Si no: puede plantarse con la evaluación estática (stand pat) o jugar
            una jugada que crea >= 2 amenazas (obliga al rival a gastar sus 2 piedras).
        Así se ve más allá del horizonte en las secuencias de amenaza-defensa, que
        son las que deciden las partidas de Connect6.
        """
        self.m_total_nodes += 1
        self.m_qs_nodes += 1
        if self.m_deadline is not None and time.perf_counter() > self.m_deadline:
            raise SearchTimeout()
        stand = evaluate(self.m_board, ourColor, preMove, features=self.m_feat)
        if check_game_over(self.m_board, preMove) is not None or abs(stand) >= WIN_SCORE_QS:
            return stand                              # terminal o táctica ya resuelta
        is_max = mover == ourColor
        threatened = bool(self.m_feat.threat_idx[mover ^ 3])
        if qply >= QS_MAX_PLY:
            return stand
        moves = generate_moves(self.m_board, mover, max_moves=QS_MAX_MOVES)
        if not threatened:
            # Plantarse: el que mueve siempre puede hacer una jugada tranquila
            if is_max:
                if stand >= beta:
                    return stand
                alpha = max(alpha, stand)
            else:
                if stand <= alpha:
                    return stand
                beta = min(beta, stand)
            moves = [m for m in moves if m.score >= FORCING_PRIORITY]
            if not moves:
                return stand
            best = stand
        else:
            if not moves:
                return stand
            best = Defines.MININT if is_max else Defines.MAXINT
        for move in moves:
            self._make(move, mover)
            try:
                score = self._quiescence(alpha, beta, ourColor, mover ^ 3, move, qply + 1)
            finally:
                self._unmake(move)
            if is_max:
                if score > best:
                    best = score
                alpha = max(alpha, best)
            else:
                if score < best:
                    best = score
                beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    def _order_moves(self, moves, is_root, tt_move, ply, color):
        """
        Orden de exploración de las candidatas de un nodo:
          - Raíz: por el valor que obtuvieron en la iteración anterior (la mejor,
            la variante principal, primero). Sin datos: la de la iteración anterior.
          - Resto: 1) la jugada de la tabla de transposición, 2) las jugadas
            asesinas de este nivel, 3) el resto, dentro de cada clase del
            generador (ganar / defender / amenazas dobles / normal), por la
            heurística histórica. Nunca se mezclan clases: la prioridad táctica
            del generador manda.
        """
        if is_root:
            if self.m_use_root_order and self.m_root_scores:
                rs = self.m_root_scores
                pv = move_cells(self.m_pv_move) if self.m_pv_move is not None else None
                # (es la PV, valor previo): a igualdad de valor, la PV va primera
                return sorted(moves, key=lambda m: (move_cells(m) == pv,
                                                    rs.get(move_cells(m), Defines.MININT - 1)),
                              reverse=True)
            first = move_cells(self.m_pv_move) if self.m_pv_move is not None else None
        else:
            first = tt_move
        front, rest = [], []
        killers = self.m_killers.get(ply, ()) if self.m_use_killers else ()
        for m in moves:
            cells = move_cells(m)
            if cells == first:
                front.insert(0, m)
            elif cells in killers:
                front.append(m)
            else:
                rest.append(m)
        if self.m_use_history and self.m_history:
            h = self.m_history
            rest.sort(key=lambda m: (m.score // 10**7, h.get((color, move_cells(m)), 0)), reverse=True)
        return front + rest

    def _record_cutoff(self, move, ply, depth, color):
        """Una jugada que provoca un corte es buena en posiciones parecidas:
        se recuerda como asesina de su nivel y suma 2^prof. en el historial."""
        cells = move_cells(move)
        if self.m_use_killers:
            k = self.m_killers.setdefault(ply, [])
            if cells not in k:
                k.insert(0, cells)
                del k[2:]
        if self.m_use_history:
            key = (color, cells)
            self.m_history[key] = self.m_history.get(key, 0) + (1 << depth)

    def alpha_beta_search(self, depth, alpha, beta, ourColor, bestMove, preMove):
        # 1. Contabilizamos el nodo para las métricas de rendimiento
        self.m_total_nodes += 1

        # 1b. Control de tiempo: si se ha pasado el plazo, abortamos la iteración
        #     entera (su resultado está incompleto y no es fiable).
        if self.m_deadline is not None and time.perf_counter() > self.m_deadline:
            raise SearchTimeout()

        # 2. Determinamos de quién es el turno
        is_max_node = (self.m_alphabeta_depth - depth) % 2 == 0
        current_color = ourColor if is_max_node else (ourColor ^ 3)
        is_root = depth == self.m_alphabeta_depth

        # 2b. TABLA DE TRANSPOSICIÓN: ¿hemos visto ya esta posición (con el mismo
        #     jugador al turno)? Si se buscó al menos a esta profundidad, su valor
        #     nos sirve sin volver a buscar. En la raíz no se corta: hace falta la jugada.
        key = self.m_hash ^ ZOBRIST_SIDE[current_color]
        tt_move = None
        if self.m_use_tt:
            entry = self.m_tt.get(key)
            if entry is not None:
                e_depth, e_value, e_flag, tt_move = entry
                if e_depth >= depth and not is_root:
                    if e_flag == EXACT:
                        self.m_tt_hits += 1
                        return e_value
                    if e_flag == LOWER:           # el valor real es >= e_value
                        alpha = max(alpha, e_value)
                    else:                         # UPPER: el valor real es <= e_value
                        beta = min(beta, e_value)
                    if alpha >= beta:
                        self.m_tt_hits += 1
                        return e_value
        # Ventana con la que se busca DE VERDAD este nodo (ya ajustada con la
        # tabla): sirve para saber después si el valor obtenido es exacto o una cota.
        alpha_orig, beta_orig = alpha, beta

        # 3. Condición de parada (límite de profundidad o fin del juego)
        terminal = check_game_over(self.m_board, preMove) is not None
        if depth == 0 and not terminal and self._needs_quiescence(current_color):
            # Hoja NO tranquila: el que mueve está amenazado -> se sigue buscando
            # solo con jugadas forzadas (no se guarda en la tabla: su valor
            # depende de la ventana alfa-beta).
            return self._quiescence(alpha, beta, ourColor, current_color, preMove, 0)
        if depth == 0 or terminal:
            value = evaluate(self.m_board, ourColor, preMove, features=self.m_feat)
            if self.m_use_tt:
                # Una posición terminal vale lo mismo a cualquier profundidad
                self.m_tt[key] = (0 if depth == 0 else 99, value, EXACT, None)
            return value

        # 4. Generar jugadas candidatas
        moves = generate_moves(self.m_board, current_color, max_moves=MAX_MOVES)

        # 4b. Ordenación (cuanto antes aparezca la mejor jugada, más poda alfa-beta)
        ply = self.m_alphabeta_depth - depth
        moves = self._order_moves(moves, is_root, tt_move, ply, current_color)

        best_cells = None

        # --- NODO MAX (Nuestro turno: buscamos maximizar la puntuación) ---
        if is_max_node:
            best_score = Defines.MININT
            # En la raíz, partimos de la primera candidata: si todas valen MININT
            # (derrota inevitable) bestMove nunca se actualizaría y devolveríamos
            # la jugada previa del rival -> jugada ilegal.
            if is_root and moves:
                self._copy_move(moves[0], bestMove)
            for move in moves:
                k = move_key(move, current_color)
                self._make(move, current_color)
                self.m_hash ^= k
                try:
                    score = self.alpha_beta_search(depth - 1, alpha, beta, ourColor, bestMove, move)
                finally:
                    self._unmake(move)
                    self.m_hash ^= k

                if score > best_score:
                    best_score = score
                    best_cells = move_cells(move)
                    # Solo guardamos el movimiento real si estamos en la raíz del árbol
                    if is_root:
                        self._copy_move(move, bestMove)
                if is_root:
                    self.m_root_scores_cur[move_cells(move)] = score

                # PODA ALFA-BETA: Actualizamos alfa y podamos si es necesario
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    self._record_cutoff(move, ply, depth, current_color)
                    break # El oponente nunca nos dejará llegar a esta rama, dejamos de explorar

        # --- NODO MIN (Turno del rival: busca minimizar nuestra puntuación) ---
        else:
            best_score = Defines.MAXINT
            for move in moves:
                k = move_key(move, current_color)
                self._make(move, current_color)
                self.m_hash ^= k
                try:
                    score = self.alpha_beta_search(depth - 1, alpha, beta, ourColor, bestMove, move)
                finally:
                    self._unmake(move)
                    self.m_hash ^= k

                if score < best_score:
                    best_score = score
                    best_cells = move_cells(move)

                # PODA ALFA-BETA: Actualizamos beta y podamos si es necesario
                beta = min(beta, best_score)
                if beta <= alpha:
                    self._record_cutoff(move, ply, depth, current_color)
                    break # Nosotros nunca elegiremos llegar a esta rama, dejamos de explorar

        # 5. Guardar en la tabla. Con alfa-beta el valor solo es exacto si cayó
        #    dentro de la ventana de búsqueda; si no, es una cota:
        #      <= alfa -> cota superior (UPPER): ninguna jugada llegó a alfa
        #      >= beta -> cota inferior (LOWER): hubo un corte beta
        if self.m_use_tt:
            if best_score <= alpha_orig:
                flag = UPPER
            elif best_score >= beta_orig:
                flag = LOWER
            else:
                flag = EXACT
            self.m_tt[key] = (depth, best_score, flag, best_cells)
        return best_score

    # -----------------------------------------------------------------------
    # Profundización iterativa con límite de tiempo (sesión 4)
    # -----------------------------------------------------------------------
    def iterative_deepening(self, board, ourColor, bestMove, preMove, time_limit,
                            max_depth=MAX_ID_DEPTH):
        """
        Alfa-beta completo a profundidad 1, 2, 3, ... hasta agotar time_limit (s).
        Devuelve el valor de la última iteración COMPLETADA y deja su jugada en
        bestMove. Una iteración interrumpida por tiempo se descarta entera.
        """
        start = time.perf_counter()
        self.before_search(board, ourColor, 1, deadline=start + time_limit)
        # game_engine pasa el mismo objeto como bestMove y preMove: copiamos la
        # jugada previa para que escribir bestMove entre iteraciones no la pise.
        pre = StoneMove()
        self._copy_move(preMove, pre)
        preMove = pre
        self.m_depth_reached = 0
        self.m_iterations = []

        # Jugada única (victoria inmediata o única defensa posible): no hay nada que buscar
        root_moves = generate_moves(self.m_board, ourColor, max_moves=MAX_MOVES)
        if len(root_moves) == 1:
            self._copy_move(root_moves[0], bestMove)
            self.m_iterations.append((0, 0, time.perf_counter() - start, None))
            return 0

        best_score = None
        prev_iter_time = None
        for depth in range(1, max_depth + 1):
            t0 = time.perf_counter()
            nodes0 = self.m_total_nodes
            self.m_alphabeta_depth = depth
            candidate = StoneMove()
            self.m_root_scores_cur = {}
            try:
                score = self.alpha_beta_search(depth, Defines.MININT, Defines.MAXINT,
                                               ourColor, candidate, preMove)
            except SearchTimeout:
                break                              # nos quedamos con la iteración anterior
            t_iter = time.perf_counter() - t0
            self.m_iterations.append((depth, self.m_total_nodes - nodes0, t_iter, score))

            # Si a esta profundidad TODAS las jugadas pierden, mantenemos la elegida
            # en la iteración anterior: con menos profundidad no veía la derrota,
            # así que es la que más la retrasa (y el rival aún puede fallar).
            if best_score is not None and score <= Defines.MININT:
                break

            self._copy_move(candidate, bestMove)
            self.m_pv_move = candidate
            self.m_root_scores = self.m_root_scores_cur
            best_score = score
            self.m_depth_reached = depth

            # Victoria (o derrota) segura ya demostrada: profundizar no cambia nada
            if abs(score) >= Defines.MAXINT:
                break

            # ¿Da tiempo a la siguiente iteración? Se estima su duración con el
            # factor de ramificación efectivo medido (t_d / t_{d-1}); si no cabe en
            # lo que queda, no se empieza (sería tiempo tirado: se abortaría).
            ebf = t_iter / prev_iter_time if prev_iter_time else DEFAULT_EBF
            ebf = max(ebf, 2.0)
            remaining = self.m_deadline - time.perf_counter()
            if t_iter * ebf > remaining:
                break
            prev_iter_time = max(t_iter, 1e-4)

        return best_score if best_score is not None else 0

def flush_output():
    import sys
    sys.stdout.flush()
