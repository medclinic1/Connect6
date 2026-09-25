from tools import *
from itertools import combinations

def get_candidate_cells(board, radius=2):
    """
    Casillas vacías a distancia <= radius (en cualquier dirección) de alguna piedra.
    Devuelve un diccionario {(x, y): nº de piedras vecinas}.
    """
    candidates = {}
    for i in range(1, Defines.GRID_NUM - 1):
        for j in range(1, Defines.GRID_NUM - 1):
            if board[i][j] not in (Defines.BLACK, Defines.WHITE):
                continue
            # Recorremos el cuadrado (2*radius+1)x(2*radius+1) alrededor de la piedra
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    x, y = i + di, j + dj
                    if isValidPos(x, y) and board[x][y] == Defines.NOSTONE:
                        # Cada piedra cercana suma 1 a la casilla vacía
                        candidates[(x, y)] = candidates.get((x, y), 0) + 1
    return candidates


def generate_moves(board, color, max_moves=20, max_cells=10, radius=2):
    """
    Genera jugadas candidatas (StoneMove de 2 piedras) para 'color'.
      - max_moves: nº máximo de jugadas devueltas (factor de ramificación).
      - max_cells: nº de casillas con las que se forman las parejas.
      - radius:    distancia máxima a una piedra existente.
    'color' no se usa en esta versión naive; se mantiene en la firma
    porque la versión mejorada (sesión 2) analizará amenazas propias y rivales.
    """
    center = (Defines.GRID_NUM - 1) // 2
    candidates = get_candidate_cells(board, radius)

    # Tablero vacío: la única jugada razonable es el centro (1 piedra)
    if not candidates:
        m = StoneMove()
        m.positions[0].x = m.positions[1].x = center
        m.positions[0].y = m.positions[1].y = center
        return [m]

    # Nos quedamos con las max_cells casillas con más piedras alrededor
    cells = sorted(candidates, key=lambda c: candidates[c], reverse=True)[:max_cells]

    # Caso extremo: menos de 2 casillas cerca -> completamos con cualquier vacía
    if len(cells) < 2:
        cells += [(i, j) for i in range(1, Defines.GRID_NUM - 1)
                         for j in range(1, Defines.GRID_NUM - 1)
                  if board[i][j] == Defines.NOSTONE and (i, j) not in cells][:2 - len(cells)]

    # Todas las parejas de casillas -> una jugada por pareja
    moves = []
    for (x1, y1), (x2, y2) in combinations(cells, 2):
        m = StoneMove()
        m.positions[0].x, m.positions[0].y = x1, y1
        m.positions[1].x, m.positions[1].y = x2, y2
        m.score = candidates.get((x1, y1), 0) + candidates.get((x2, y2), 0)
        moves.append(m)

    # Ordenamos de más a menos prometedora y limitamos
    moves.sort(key=lambda m: m.score, reverse=True)
    return moves[:max_moves]