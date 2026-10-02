from tools import *
from itertools import combinations
from evaluation import WINDOWS

def get_evaluated_cells(board, current_color):
    """
    Analiza las amenazas de ataque y defensa para priorizar casillas vacías.
    Asigna una puntuación a cada casilla vacía según la severidad de la amenaza.
    """
    cell_scores = {}
    rival_color = current_color ^ 3
    
    # Pesos exponenciales según la severidad (nº de fichas en una ventana viva)
    # 5 fichas -> victoria inminente (prioridad máxima)
    # 4 fichas -> amenaza grave (prioridad alta)
    # 3 fichas -> amenaza media
    # 2, 1 -> desarrollo de tablero
    weights = {5: 100000, 4: 10000, 3: 1000, 2: 100, 1: 10, 0: 0}
    
    # Analizamos todas las ventanas de 6 que caben en el tablero
    for w in WINDOWS:
        vals = [board[x][y] for x, y in w]
        mine = vals.count(current_color)
        theirs = vals.count(rival_color)
        
        # Solo proporcionamos candidatos en grupos "vivos" (sin mezclar colores)
        if mine > 0 and theirs == 0:
            score = weights[mine]
            for x, y in w:
                if board[x][y] == Defines.NOSTONE:
                    cell_scores[(x, y)] = cell_scores.get((x, y), 0) + score
                    
        elif theirs > 0 and mine == 0:
            # Amenazas defensivas (priorizamos ligeramente el ataque en caso de empate)
            score = weights[theirs] * 0.9 
            for x, y in w:
                if board[x][y] == Defines.NOSTONE:
                    cell_scores[(x, y)] = cell_scores.get((x, y), 0) + score
                    
    return cell_scores

def generate_moves(board, current_color, max_moves=20, max_cells=12):
    center = (Defines.GRID_NUM - 1) // 2
    
    # 1. Obtener casillas priorizadas por amenazas
    cell_scores = get_evaluated_cells(board, current_color)
    
    # Tablero inicial vacío
    if not cell_scores:
        m = StoneMove()
        m.positions[0].x = m.positions[1].x = center
        m.positions[0].y = m.positions[1].y = center
        return [m]
        
    # 2. Seleccionar solo las casillas más críticas
    sorted_cells = sorted(cell_scores.keys(), key=lambda c: cell_scores[c], reverse=True)
    best_cells = sorted_cells[:max_cells]
    
    if len(best_cells) < 2:
        for i in range(1, Defines.GRID_NUM - 1):
            for j in range(1, Defines.GRID_NUM - 1):
                if board[i][j] == Defines.NOSTONE and (i, j) not in best_cells:
                    best_cells.append((i, j))
                    if len(best_cells) >= 2: break
                    
    # 3. Generar y puntuar combinaciones de pares de piedras
    moves = []
    for (x1, y1), (x2, y2) in combinations(best_cells, 2):
        m = StoneMove()
        m.positions[0].x, m.positions[0].y = x1, y1
        m.positions[1].x, m.positions[1].y = x2, y2
        
        # Sumamos la severidad de las dos casillas implicadas
        m.score = cell_scores.get((x1, y1), 0) + cell_scores.get((x2, y2), 0)
        moves.append(m)

    # 4. Ordenar los candidatos del mejor al peor
    moves.sort(key=lambda m: m.score, reverse=True)
    return moves[:max_moves]