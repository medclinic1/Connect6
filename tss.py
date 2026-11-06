"""
Oráculo de amenazas (sesión 5): demuestra victorias forzadas en Connect6.
Lo usa el motor en la raíz, antes de alfa-beta (search_engine.py, USE_TSS):
juega las victorias demostradas y descarta las jugadas que pierden. También
sirve para etiquetar las posiciones de la batería de pruebas (bateria.py).

Idea (búsqueda en el espacio de amenazas, Wu y Huang 2005): el atacante solo
juega jugadas que dejan al rival sin elección.
  - Amenaza = ventana viva del atacante con >= 4 piedras (se completa en 1 turno).
  - T = nº mínimo de piedras que necesita el defensor para tapar todas
    (count_threats: 0, 1, 2 o 3 = "imposible con 2 piedras").
  - Victoria en 1 turno: el atacante ya tiene una amenaza.
  - Victoria en 2: una jugada deja T >= 3 (el defensor no puede tapar todo).
  - Victoria en k >= 3: una jugada deja T = 2 (el defensor gasta sus dos
    piedras en tapar) y, para TODAS las formas de tapar, hay victoria en k-1.
Es CORRECTO (lo que demuestra es una victoria real: se revisan todas las
defensas) pero INCOMPLETO: no considera jugadas con T = 1 (el defensor tiene
una piedra libre) ni jugadas tranquilas. Por eso sirve para etiquetar
"victoria forzada en <= k turnos", no para negar que exista.
Cada jugada del atacante también tiene que tapar las amenazas del defensor
(si no, el defensor gana antes).
"""
import time
from itertools import combinations
from defines import Defines
from evaluation import WINDOWS, extract_features, count_threats
from move_generator import CELL_WINDOWS


class TSSTimeout(Exception):
    """Se agotó el tiempo concedido al oráculo: la pregunta queda sin demostrar."""
    pass


_deadline = None            # instante (perf_counter) límite; None = sin límite


def set_deadline(deadline):
    global _deadline
    _deadline = deadline


def _check_time():
    if _deadline is not None and time.perf_counter() > _deadline:
        raise TSSTimeout()


def threat_windows(board, color):
    return extract_features(board)[2][color]


def _live_windows(board, color):
    """Una pasada por las 924 ventanas: las vivas de 'color' con >= 2 piedras,
    como {índice: (nº de piedras, huecos)}, y para cada hueco sus ventanas.
    Solo en ellas puede aparecer una amenaza (>= 4 piedras) añadiendo <= 2."""
    rival = color ^ 3
    live, by_cell = {}, {}
    for idx, w in enumerate(WINDOWS):
        vals = [board[x][y] for x, y in w]
        n = vals.count(color)
        if n < 2 or rival in vals:
            continue
        empties = frozenset(c for c, v in zip(w, vals) if v == Defines.NOSTONE)
        live[idx] = (n, empties)
        for c in empties:
            by_cell.setdefault(c, []).append(idx)
    return live, by_cell


def _pair_threats(live, by_cell, c1, c2):
    """Ventanas amenazantes (como huecos que quedan) tras poner c1 y c2.
    El atacante no tenía amenazas antes, así que son todas las que hay."""
    out = []
    for idx in set(by_cell.get(c1, ())) | set(by_cell.get(c2, ())):
        n, empties = live[idx]
        n += (c1 in empties) + (c2 in empties)
        if n >= 4:
            out.append(empties - {c1, c2})
    return out


def attacking_moves(board, color):
    """Jugadas (c1, c2) que tapan todas las amenazas del rival y crean T >= 2.
    Devuelve [(T, c1, c2, ventanas)] ordenadas de más a menos amenazas.
    Solo se prueban parejas que pueden crear amenazas: casillas de una ventana
    con 3 piedras (basta 1 piedra) o dos casillas de la misma ventana con 2."""
    rival_thr = threat_windows(board, color ^ 3)
    live, by_cell = _live_windows(board, color)
    cells = sorted(by_cell)
    three = {c for c in cells if any(live[i][0] >= 3 for i in by_cell[c])}
    pairs = set()
    for c in three:                        # una piedra ya hace amenaza: la otra, donde sea
        for d in cells:
            if d != c:
                pairs.add((min(c, d), max(c, d)))
    for n, empties in live.values():       # las dos piedras en la misma ventana
        if n == 2:
            for a, b in combinations(sorted(empties), 2):
                pairs.add((a, b))
    res = []
    for c1, c2 in pairs:
        if any(c1 not in w and c2 not in w for w in rival_thr):
            continue                       # deja una amenaza rival viva: perdería
        thr = _pair_threats(live, by_cell, c1, c2)
        if len(thr) < 2:
            continue
        t = count_threats(thr)
        if t >= 2:
            res.append((t, c1, c2, thr))
    res.sort(key=lambda r: (-r[0], r[1], r[2]))
    return res


def blocks(board, windows):
    """Todas las parejas de casillas vacías que tocan todas las ventanas."""
    union = sorted(frozenset.union(*windows))
    return [(a, b) for a, b in combinations(union, 2)
            if all(a in w or b in w for w in windows)]


def win_in(board, color, k, line=None):
    """¿'color' (al turno) gana a la fuerza en <= k turnos suyos?
    Si line es una lista, se rellena con la primera jugada ganadora."""
    if k < 1:
        return False
    own = threat_windows(board, color)
    if own:
        if line is not None:
            line.append(("gana", sorted(min(own, key=len))))
        return True
    if k < 2:
        return False
    for t, c1, c2, thr in attacking_moves(board, color):
        _check_time()
        if t >= 3:
            if line is not None:
                line.append((c1, c2))
            return True
        if k < 3:
            continue
        # t == 2: el defensor tiene que tapar con sus dos piedras
        board[c1[0]][c1[1]] = board[c2[0]][c2[1]] = color
        ok = True
        try:
            for d1, d2 in blocks(board, thr):
                board[d1[0]][d1[1]] = board[d2[0]][d2[1]] = color ^ 3
                try:
                    sub = win_in(board, color, k - 1)
                finally:
                    board[d1[0]][d1[1]] = board[d2[0]][d2[1]] = Defines.NOSTONE
                if not sub:
                    ok = False
                    break
        finally:
            board[c1[0]][c1[1]] = board[c2[0]][c2[1]] = Defines.NOSTONE
        if ok:
            if line is not None:
                line.append((c1, c2))
            return True
    return False


def min_win(board, color, kmax=3):
    """Menor k <= kmax con victoria forzada demostrada (None si no se demuestra)."""
    for k in range(1, kmax + 1):
        if win_in(board, color, k):
            return k
    return None
