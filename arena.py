"""
Banco de pruebas (sesión 4): enfrenta dos versiones del motor a través de su
interfaz de texto, igual que la GUI, desde un conjunto de aperturas y
alternando colores.

Por qué hace falta: el motor es determinista. Dos versiones que empiezan
desde el tablero vacío juegan siempre la misma partida, así que 1 partida
no dice nada. Con N aperturas distintas y jugando cada una con los dos
colores se obtienen 2N partidas diferentes y se compensa la ventaja de salida.

Uso:
  python arena.py DIR_A DIR_B [--openings N] [--time T] [--seed S]
     DIR_A / DIR_B : carpetas con main.py de cada versión
     --time T      : se manda "depth T" a cada motor (tiempo o profundidad,
                     según lo interprete esa versión)
"""
import argparse
import os
import random
import re
import subprocess
import sys
import time

from defines import Defines, StoneMove
from tools import init_board, make_move, move2msg, msg2move, check_game_over

CENTER = (Defines.GRID_NUM - 1) // 2      # (10, 10) = JJ
MAX_TURNS = 120                            # tope de jugadas: si se alcanza, tablas


class EngineProcess:
    """Un motor en un subproceso, hablado por stdin/stdout como hace la GUI."""

    def __init__(self, folder, depth=None, extra_cmds=()):
        self.folder = folder
        self.proc = subprocess.Popen([sys.executable, "main.py"], cwd=folder,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        if depth is not None:
            self.send(f"depth {depth}")
        for c in extra_cmds:
            self.send(c)

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_move(self):
        """Lee hasta la línea 'move XXXX'. Devuelve (jugada, info) con las métricas impresas."""
        info = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"El motor de {self.folder} se ha cerrado")
            line = line.strip()
            # Ojo: la ayuda que imprime el motor al arrancar también tiene una
            # línea " move XXXX   - tell the engine...": solo vale "move" + 2 o 4 letras.
            if re.fullmatch(r"move [A-S]{2}([A-S]{2})?", line):
                return line[5:], info
            for key in ("AB Time", "Depth", "Node", "Score"):
                if line.startswith(key + ":"):
                    try:
                        info[key] = float(line.split(":", 1)[1])
                    except ValueError:
                        pass

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def make_openings(n, seed=0, radius=2):
    """
    n aperturas reproducibles: negro en JJ, blanco 2 piedras y (en la mitad de
    ellas) negro otras 2, todas a distancia <= radius del centro.
    Cada apertura es una lista de (color, "XXXX").
    """
    rng = random.Random(seed)
    cells = [(CENTER + dx, CENTER + dy) for dx in range(-radius, radius + 1)
             for dy in range(-radius, radius + 1) if (dx, dy) != (0, 0)]
    openings, seen = [], set()
    while len(openings) < n:
        picks = rng.sample(cells, 4)
        with_black2 = len(openings) % 2 == 1
        key = (frozenset(picks[:2]), frozenset(picks[2:]) if with_black2 else None)
        if key in seen:
            continue
        seen.add(key)
        seq = [(Defines.BLACK, _msg([(CENTER, CENTER)])),
               (Defines.WHITE, _msg(picks[:2]))]
        if with_black2:
            seq.append((Defines.BLACK, _msg(picks[2:])))
        openings.append(seq)
    return openings


def _msg(cells):
    m = StoneMove()
    (x1, y1), (x2, y2) = cells[0], cells[-1]
    m.positions[0].x, m.positions[0].y = x1, y1
    m.positions[1].x, m.positions[1].y = x2, y2
    return move2msg(m)


def play_game(dir_black, dir_white, opening, depth=None, verbose=False):
    """
    Juega una partida. Devuelve (ganador, nº de jugadas, métricas por color).
    ganador: Defines.BLACK / Defines.WHITE / Defines.DRAW.
    Una jugada ilegal (casilla ocupada o fuera) pierde la partida.
    """
    board = [[0] * Defines.GRID_NUM for _ in range(Defines.GRID_NUM)]
    init_board(board)
    engines = {Defines.BLACK: EngineProcess(dir_black, depth),
               Defines.WHITE: EngineProcess(dir_white, depth)}
    stats = {Defines.BLACK: [], Defines.WHITE: []}
    try:
        # 1. Colocar la apertura en el tablero propio y en los dos motores
        last = None
        for color, msg in opening:
            make_move(board, msg2move(msg), color)
            word = "black" if color == Defines.BLACK else "white"
            for e in engines.values():
                e.send(f"{word} {msg}")
            last = color
        mover = last ^ 3
        # 2. El que mueve pide "next"; a partir de ahí, "move XXXX" al otro
        engines[mover].send("next")
        turns = len(opening)
        while True:
            msg, info = engines[mover].wait_move()
            stats[mover].append(info)
            move = msg2move(msg)
            for p in move.positions:
                if not (0 < p.x < Defines.GRID_NUM - 1 and 0 < p.y < Defines.GRID_NUM - 1) \
                        or board[p.x][p.y] != Defines.NOSTONE:
                    if verbose:
                        print(f"  jugada ilegal {msg} de {mover}")
                    return mover ^ 3, turns, stats
            make_move(board, move, mover)
            turns += 1
            result = check_game_over(board, move)
            if result is not None:
                return result, turns, stats
            if turns >= MAX_TURNS:
                return Defines.DRAW, turns, stats
            mover ^= 3
            engines[mover].send(f"move {msg}")
    finally:
        for e in engines.values():
            e.close()


def _play_pair(args):
    i, op, a_color, dir_a, dir_b, depth = args
    black, white = (dir_a, dir_b) if a_color == Defines.BLACK else (dir_b, dir_a)
    t0 = time.perf_counter()
    winner, turns, stats = play_game(black, white, op, depth)
    return i, a_color, winner, turns, stats, time.perf_counter() - t0


def match(dir_a, dir_b, n_openings=8, depth=None, seed=0, verbose=True, workers=1):
    """Cada apertura dos veces (A con negras y A con blancas). Devuelve resumen.
    workers > 1 juega varias partidas a la vez (cada motor piensa solo en su
    turno, así que con 4 núcleos caben 4 partidas sin quitarse CPU)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(i, op, a_color, dir_a, dir_b, depth)
            for i, op in enumerate(make_openings(n_openings, seed))
            for a_color in (Defines.BLACK, Defines.WHITE)]
    res = {"A": 0, "B": 0, "draw": 0}
    times = {"A": [], "B": [], "depthA": [], "depthB": []}
    turns_list = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, a_color, winner, turns, stats, dt in pool.map(_play_pair, jobs):
            turns_list.append(turns)
            if winner == Defines.DRAW:
                res["draw"] += 1
                tag = "tablas"
            elif winner == a_color:
                res["A"] += 1
                tag = "gana A"
            else:
                res["B"] += 1
                tag = "gana B"
            for color, who in ((a_color, "A"), (a_color ^ 3, "B")):
                for s in stats[color]:
                    if "AB Time" in s:
                        times[who].append(s["AB Time"])
                    if "Depth" in s:
                        times["depth" + who].append(s["Depth"])
            if verbose:
                print(f"apertura {i:2d}  A={'negras' if a_color == 1 else 'blancas'}  "
                      f"{tag:7s} en {turns:3d} jugadas  ({dt:.1f} s)", flush=True)
    summary = dict(res)
    for who in ("A", "B"):
        t = times[who]
        summary[f"t_medio_{who}"] = round(sum(t) / len(t), 3) if t else None
        summary[f"t_max_{who}"] = max(t) if t else None
        d = times["depth" + who]
        summary[f"prof_media_{who}"] = round(sum(d) / len(d), 2) if d else None
    summary["jugadas_media"] = sum(turns_list) / len(turns_list)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--openings", type=int, default=8)
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    depth = None if args.time is None else (int(args.time) if args.time.is_integer() else args.time)
    s = match(os.path.abspath(args.dir_a), os.path.abspath(args.dir_b),
              args.openings, depth, args.seed, workers=args.workers)
    print(s)
