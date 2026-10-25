from tools import *
from itertools import combinations
from evaluation import WINDOWS, THREAT_STONES, count_threats

# ---------------------------------------------------------------------------
# Índice casilla -> ventanas que la contienen (se calcula una vez al importar).
# Permite saber qué ventanas cambian al colocar una piedra sin recorrer las 924.
# ---------------------------------------------------------------------------
CELL_WINDOWS = {}
for _idx, _w in enumerate(WINDOWS):
    for _c in _w:
        CELL_WINDOWS.setdefault(_c, []).append(_idx)

# Pesos exponenciales según la severidad (nº de fichas en una ventana viva)
# 5 fichas -> victoria inminente (prioridad máxima)
# 4 fichas -> amenaza grave (prioridad alta)
# 3 fichas -> amenaza media
# 2, 1 -> desarrollo de tablero
CELL_WEIGHTS = {5: 100000, 4: 10000, 3: 1000, 2: 100, 1: 10, 0: 0}
DEFENSE_FACTOR = 0.9     # Priorizamos ligeramente el ataque en caso de empate
FORCING_THREATS = 2      # Amenazas mínimas para que una jugada pase delante de todas


def scan_board(board, current_color):
    """
    Una sola pasada por las 924 ventanas. Devuelve:
      - cell_scores : puntuación de cada casilla vacía según la severidad de las
                      ventanas vivas (propias y rivales) que la contienen
      - my_threats  : ventanas vivas propias con >= 4 piedras (frozenset de huecos)
      - rival_threats : ídem para el rival
    """
    cell_scores = {}
    my_threats, rival_threats = [], []
    rival_color = current_color ^ 3

    for w in WINDOWS:
        vals = [board[x][y] for x, y in w]
        mine = vals.count(current_color)
        theirs = vals.count(rival_color)

        # Ventana completa (6 fichas): no hay huecos que evaluar (evita KeyError)
        if mine == 6 or theirs == 6:
            continue

        # Solo proporcionamos candidatos en grupos "vivos" (sin mezclar colores)
        if mine > 0 and theirs == 0:
            score, n, threats = CELL_WEIGHTS[mine], mine, my_threats
        elif theirs > 0 and mine == 0:
            score, n, threats = CELL_WEIGHTS[theirs] * DEFENSE_FACTOR, theirs, rival_threats
        else:
            continue

        empties = [c for c, v in zip(w, vals) if v == Defines.NOSTONE]
        for c in empties:
            cell_scores[c] = cell_scores.get(c, 0) + score
        if n >= THREAT_STONES:
            threats.append(frozenset(empties))

    return cell_scores, my_threats, rival_threats


def get_evaluated_cells(board, current_color):
    """Compatibilidad con la versión de la sesión 2: solo las puntuaciones."""
    return scan_board(board, current_color)[0]


def _make_move(c1, c2, score):
    m = StoneMove()
    m.positions[0].x, m.positions[0].y = c1
    m.positions[1].x, m.positions[1].y = c2
    m.score = score
    return m


def _any_other_cell(board, cell_scores, exclude):
    """Mejor casilla vacía distinta de 'exclude' (para la 2.ª piedra de una jugada)."""
    for c in sorted(cell_scores, key=cell_scores.get, reverse=True):
        if c != exclude:
            return c
    for i in range(1, Defines.GRID_NUM - 1):
        for j in range(1, Defines.GRID_NUM - 1):
            if board[i][j] == Defines.NOSTONE and (i, j) != exclude:
                return (i, j)
    return exclude


def threats_created(board, color, c1, c2):
    """
    Nº de amenazas (0, 1, 2 o 3 = 'imparables') que tendría 'color' tras jugar c1 y c2.
    Solo revisa las ventanas que contienen c1 o c2 (las únicas que cambian), por lo
    que es mucho más barato que volver a analizar el tablero.
    """
    rival = color ^ 3
    windows = []
    for idx in set(CELL_WINDOWS[c1]) | set(CELL_WINDOWS[c2]):
        w = WINDOWS[idx]
        vals = [board[x][y] for x, y in w]
        if rival in vals:
            continue                               # ventana muerta
        mine = vals.count(color) + (c1 in w) + (c2 in w)
        if mine >= THREAT_STONES:
            windows.append(frozenset(c for c, v in zip(w, vals)
                                     if v == Defines.NOSTONE and c != c1 and c != c2))
    return count_threats(windows)


def generate_moves(board, current_color, max_moves=20, max_cells=12):
    center = (Defines.GRID_NUM - 1) // 2

    # 0. Tablero físicamente vacío (apertura real): una piedra en el centro.
    #    No basta con "no hay ventanas vivas": al final de la partida puede no
    #    quedar ninguna y se devolvería JJ (ocupada) -> jugada ilegal.
    if all(board[i][j] == Defines.NOSTONE
           for i in range(1, Defines.GRID_NUM - 1) for j in range(1, Defines.GRID_NUM - 1)):
        m = StoneMove()
        m.positions[0].x = m.positions[1].x = center
        m.positions[0].y = m.positions[1].y = center
        return [m]

    cell_scores, my_threats, rival_threats = scan_board(board, current_color)

    # 1. ¿Podemos ganar en este turno? Una ventana viva propia con >= 4 piedras
    #    tiene <= 2 huecos: rellenarlos es victoria. Es la única jugada necesaria.
    if my_threats:
        gaps = sorted(min(my_threats, key=len))
        c1 = gaps[0]
        c2 = gaps[1] if len(gaps) > 1 else _any_other_cell(board, cell_scores, c1)
        return [_make_move(c1, c2, Defines.MAXINT)]

    # 2. ¿Nos amenaza el rival con ganar en su próximo turno? Solo proponemos
    #    defensas: parejas de casillas que tocan (bloquean) sus ventanas amenazantes.
    if rival_threats:
        n_threats = count_threats(rival_threats)
        defense_cells = sorted(frozenset.union(*rival_threats),
                               key=cell_scores.get, reverse=True)
        # Para la 2.ª piedra también valen las mejores casillas generales
        # (con una amenaza basta una piedra para defender y la otra puede atacar).
        others = sorted(cell_scores, key=cell_scores.get, reverse=True)[:max_cells]
        pool = list(dict.fromkeys(defense_cells + others))

        moves = []
        for c1, c2 in combinations(pool, 2):
            blocked = sum(1 for w in rival_threats if c1 in w or c2 in w)
            if n_threats <= 2 and blocked < len(rival_threats):
                continue                       # no bloquea todo -> perdemos seguro
            score = cell_scores[c1] + cell_scores[c2]
            # Orden: más ventanas bloqueadas primero; después, mejor puntuación
            moves.append(_make_move(c1, c2, blocked * 10**7 + score))
        moves.sort(key=lambda m: m.score, reverse=True)
        return moves[:max_moves]

    # 3. Posición sin amenazas inmediatas: casillas más prometedoras ...
    best_cells = sorted(cell_scores, key=cell_scores.get, reverse=True)[:max_cells]

    #    Tablero bloqueado (casi sin ventanas vivas): rellenar con casillas vacías
    if len(best_cells) < 2:
        for i in range(1, Defines.GRID_NUM - 1):
            for j in range(1, Defines.GRID_NUM - 1):
                if board[i][j] == Defines.NOSTONE and (i, j) not in best_cells:
                    best_cells.append((i, j))
                    if len(best_cells) >= max_cells:
                        break
            if len(best_cells) >= max_cells:
                break

    # 4. ... combinadas en parejas y ordenadas primero por la FUERZA DE LA AMENAZA
    #    que crean (3 = victoria forzada, 2 = obliga a gastar las 2 piedras rivales,
    #    1 = obliga a gastar 1) y, a igualdad, por la puntuación de las casillas.
    moves = []
    for c1, c2 in combinations(best_cells, 2):
        t = threats_created(board, current_color, c1, c2)
        score = cell_scores.get(c1, 0) + cell_scores.get(c2, 0)
        priority = t if t >= FORCING_THREATS else 0
        moves.append(_make_move(c1, c2, priority * 10**7 + score))

    moves.sort(key=lambda m: m.score, reverse=True)
    return moves[:max_moves]
