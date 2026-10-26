"""
Batería de pruebas de la sesión 5.

  python bateria.py build PARTIDAS.txt [más .txt] -o bateria.json
      Recorre partidas (una por línea, jugadas en formato de la GUI) y etiqueta
      cada posición con el oráculo de amenazas (tss.py):
        G1/G2/G3  el que mueve tiene victoria forzada en exactamente k turnos
        D1/D2/D3  el que mueve no gana, el rival ganaría en k turnos si pasara,
                  y existe al menos una jugada tras la que el rival ya no tiene
                  victoria demostrada en <= k (la posición se puede defender)
      Se añaden también las aperturas de arena.py (categoría A).

  python bateria.py run bateria.json MOTOR CLOUDICT NIVEL -o resultados.json
      Juega cada posición con nuestro motor al turno contra Cloudict, con el
      mismo protocolo que la GUI (new xxx + todas las jugadas + next), y anota
      las métricas de la primera decisión (tiempo, profundidad, nodos, nodos
      podados, cortes) y si se cumple el objetivo:
        Gk: ganamos en <= k turnos nuestros
        Dk: Cloudict no gana en sus k turnos siguientes
        A : no perdemos en las primeras 30 jugadas (se juega hasta 60)
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from defines import Defines
from tools import init_board, make_move, msg2move, move2msg
from move_generator import generate_moves
from search_engine import move_cells
import tss

EDGE = 19


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------
def board_from(seq):
    b = [[0] * Defines.GRID_NUM for _ in range(Defines.GRID_NUM)]
    init_board(b)
    for k, s in enumerate(seq):
        make_move(b, msg2move(s), Defines.BLACK if k % 2 == 0 else Defines.WHITE)
    return b


def defensible(b, mover, k):
    rival = mover ^ 3
    for m in generate_moves(b, mover, max_moves=20):
        cells = move_cells(m)
        for x, y in cells:
            b[x][y] = mover
        lost = tss.win_in(b, rival, k)
        for x, y in cells:
            b[x][y] = Defines.NOSTONE
        if not lost:
            return move2msg(m)
    return None


def build(files, per_cat=4, n_openings=4):
    from arena import make_openings
    cats = {c: [] for c in ("G1", "G2", "G3", "D1", "D2", "D3")}
    seen = set()
    for f in files:
        for line in open(f):
            seq = line.split()
            for n in range(2, len(seq)):
                key = " ".join(seq[:n])
                if key in seen:
                    continue
                seen.add(key)
                b = board_from(seq[:n])
                mover = Defines.BLACK if n % 2 == 0 else Defines.WHITE
                a = tss.min_win(b, mover, 3)
                if a:
                    cats[f"G{a}"].append(dict(seq=key, cat=f"G{a}", mover=mover))
                    continue
                d = tss.min_win(b, mover ^ 3, 3)
                if d:
                    ok = defensible(b, mover, d)
                    if ok:
                        cats[f"D{d}"].append(dict(seq=key, cat=f"D{d}", mover=mover, defensa=ok))
    out = []
    for c, lst in cats.items():
        # las de distinto color y distinta partida primero (más variedad)
        lst.sort(key=lambda p: (len(p["seq"].split()) % 2, len(p["seq"])))
        picked, colors = [], {}
        for p in lst:
            if colors.get(p["mover"], 0) < (per_cat + 1) // 2:
                picked.append(p)
                colors[p["mover"]] = colors.get(p["mover"], 0) + 1
            if len(picked) == per_cat:
                break
        for p in lst:
            if len(picked) == per_cat:
                break
            if p not in picked:
                picked.append(p)
        out += picked
        print(f"{c}: {len(lst)} candidatas, {len(picked)} elegidas")
    for i, op in enumerate(make_openings(n_openings, seed=7)):
        seq = [m if len(m) == 4 else m * 2 for _, m in op]
        mover = Defines.BLACK if len(seq) % 2 == 0 else Defines.WHITE
        out.append(dict(seq=" ".join(seq), cat="A", mover=mover))
        # y la misma apertura con el otro color al turno no es posible sin cambiarla:
        # jugamos cada apertura una vez con el color al que le toca
    return out


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def _gui_engine(gui_dir):
    sys.path.insert(0, gui_dir)
    from engine import GameEngine, Move
    return GameEngine, Move


def six(board, x, y, c):
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            i, j = x + s * dx, y + s * dy
            while 0 <= i < EDGE and 0 <= j < EDGE and board[i][j] == c:
                n += 1
                i += s * dx
                j += s * dy
        if n >= 6:
            return True
    return False


def run_position(GameEngine, Move, pos, ours, cloud, level, timeout=30):
    seq = pos["seq"].split()
    cat = pos["cat"]
    k = int(cat[1]) if cat != "A" else None
    max_plies = 2 * k + 1 if k else 60
    board = [[0] * EDGE for _ in range(EDGE)]
    moves = []

    def place(m):
        for idx, (x, y) in enumerate(((m.x1, m.y1), (m.x2, m.y2))):
            if not (0 <= x < EDGE and 0 <= y < EDGE):
                return "invalid"
            if board[x][y]:
                if len(moves) == 0 and idx == 1 and (x, y) == (m.x1, m.y1):
                    continue
                return "dup"
            board[x][y] = m.color
        moves.append(m)
        return "win" if six(board, m.x1, m.y1, m.color) or six(board, m.x2, m.y2, m.color) else None

    for i, s in enumerate(seq):
        place(Move.fromCmd("move " + s, Move.BLACK if i % 2 == 0 else Move.WHITE))
    our = Move.BLACK if len(seq) % 2 == 0 else Move.WHITE
    eng = {our: GameEngine(), our ^ 3: GameEngine()}
    eng[our].init(ours, level, False)
    eng[our ^ 3].init(cloud, level, False)
    first, winner, why, our_turns, cl_turns = None, None, "límite", 0, 0
    try:
        for ply in range(max_plies):
            color = Move.BLACK if len(moves) % 2 == 0 else Move.WHITE
            e = eng[color]
            t0 = time.perf_counter()
            e.next(moves)
            info = {}
            while True:
                msg = e.waitForNextMsg()
                if not msg:
                    winner, why = color ^ 3, "se cerró"
                    break
                line = msg.strip()
                m = Move.fromCmd(line, color) if re.fullmatch(r"move [A-S]{2}([A-S]{2})?", line) else None
                if m is not None:
                    break
                for key in ("AB Time", "Depth", "Node", "Pruned", "Cutoffs", "Score"):
                    if line.replace(" ", "").startswith(key.replace(" ", "") + ":"):
                        try:
                            info[key] = float(line.split(":", 1)[1])
                        except ValueError:
                            pass
                if line.startswith("TSS:"):
                    info["TSS"] = line.split(":", 1)[1].strip()
            if winner is not None:
                break
            dt = time.perf_counter() - t0
            if color == our:
                our_turns += 1
                if first is None:
                    first = dict(info, wall=round(dt, 3), move=m.cmd())
            else:
                cl_turns += 1
            if dt > timeout:
                winner, why = color ^ 3, "timeout"
                break
            r = place(m)
            if r in ("invalid", "dup"):
                winner, why = color ^ 3, r
                break
            if r == "win":
                winner, why = color, "seis"
                break
    finally:
        for x in eng.values():
            x.release()
    if cat.startswith("G"):
        ok = winner == our and our_turns <= k
    elif cat.startswith("D"):
        ok = not (winner == our ^ 3 and cl_turns <= k)
    else:
        ok = not (winner == our ^ 3 and len(moves) <= 30)
    return dict(cat=cat, seq=pos["seq"], ok=ok, ganador=("nuestro" if winner == our else
                "cloudict" if winner == our ^ 3 else "ninguno"), motivo=why,
                jugadas=len(moves) - len(seq), primera=first)


def run(bat, ours, cloud, level, gui_dir):
    GameEngine, Move = _gui_engine(gui_dir)
    res = []
    for pos in bat:
        r = run_position(GameEngine, Move, pos, ours, cloud, level)
        res.append(r)
        f = r["primera"] or {}
        print(f"{r['cat']:3s} {'OK ' if r['ok'] else 'NO '} {r['ganador']:8s} {r['motivo']:8s} "
              f"+{r['jugadas']:2d} jug.  1.ª: {f.get('move')} t={f.get('wall')} prof={f.get('Depth')} "
              f"nodos={f.get('Node')} podados={f.get('Pruned')} tss={f.get('TSS')}", flush=True)
    summary = {}
    for r in res:
        s = summary.setdefault(r["cat"], [0, 0])
        s[0] += r["ok"]
        s[1] += 1
    print({c: f"{a}/{n}" for c, (a, n) in sorted(summary.items())})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "run"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--gui", default=os.path.join(HERE, "..", "Connect6GUI"))
    ap.add_argument("--per-cat", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "build":
        data = build(a.args, per_cat=a.per_cat)
    else:
        bat, ours, cloud, level = a.args
        data = run(json.load(open(bat)), os.path.abspath(ours), os.path.abspath(cloud), int(level),
                   os.path.abspath(a.gui))
    json.dump(data, open(a.out, "w"), indent=1, ensure_ascii=False)
