"""
Algoritmo genético para optimizar los pesos de la evaluación (sesión 4).

Flujo del tema 4: representación -> inicialización -> evaluación (fitness) ->
selección de padres -> cruce -> mutación -> reemplazo -> repetición.

  Cromosoma : 7 genes reales en escala logarítmica (ver GENES).
  Fitness   : liguilla entre toda la población; cada pareja juega las mismas
              4 aperturas con los dos colores a profundidad fija (selfplay.py),
              con tope de 80 jugadas (ver fitness_noise en la documentación).
              Victoria = 1, tablas = 0,5. Aperturas nuevas en cada generación.
  Selección : torneo de tamaño k.
  Cruce     : aritmético (blend) en escala log.
  Mutación  : gaussiana en escala log, gen a gen.
  Reemplazo : elitismo (los mejores pasan tal cual) + hijos.
  Parada    : nº fijo de generaciones.

Como el fitness es relativo a la población (los individuos solo compiten entre
sí), no es comparable entre generaciones. Para ver si el AG progresa, en cada
generación el mejor individuo juega además contra los pesos por defecto en
aperturas de validación fijas (no influye en la selección).

Uso: python genetic.py --pop 10 --gens 15 --out ga_run.json
"""
import argparse
import json
import math
import random
import time
from itertools import combinations
from multiprocessing import Pool

from defines import Defines
from evaluation import DEFAULT_WEIGHTS, WEIGHT_NAMES
from selfplay import play_weights
from arena import make_openings

# ---------------------------------------------------------------------------
# Representación
# ---------------------------------------------------------------------------
# La evaluación tiene dos invarianzas de escala, así que dos pesos se fijan:
#   - W1 = 1    : multiplicar todos los W_n y CONNECT por k equivale a
#                 multiplicar TEMPO y DEFENSE por k.
#   - DEFENSE = 1 : multiplicar TEMPO, DEFENSE y FORCED por k multiplica toda
#                 la evaluación tranquila por k, y alfa-beta solo compara valores
#                 (salvo la cota de ±QUIET_LIMIT).
# Genes libres (gen = log del peso; CONNECT, cuyo valor conocido es 0, usa log(1 + c)):
GENES = ["W2", "W3", "W4", "W5", "CONNECT", "TEMPO", "FORCED"]
FIXED = {"W1": 1.0, "DEFENSE": 1.0}

# Límites (en el espacio del peso) para que la mutación no se vaya a valores absurdos
BOUNDS = {"W2": (1.0, 40.0), "W3": (2.0, 300.0), "W4": (5.0, 1500.0),
          "W5": (20.0, 5000.0), "CONNECT": (0.0, 50.0), "TEMPO": (0.5, 4.0),
          "FORCED": (5.0, 5000.0)}


def _to_gene(name, w):
    return math.log1p(w) if name == "CONNECT" else math.log(w)


def _from_gene(name, g):
    return math.expm1(g) if name == "CONNECT" else math.exp(g)


def encode(weights):
    """Fenotipo (vector de 9 pesos) -> genotipo (7 genes)."""
    return [_to_gene(n, weights[WEIGHT_NAMES.index(n)]) for n in GENES]


def decode(genome):
    """Genotipo -> vector de 9 pesos en el orden de evaluation.WEIGHT_NAMES."""
    values = dict(FIXED)
    for n, g in zip(GENES, genome):
        values[n] = _from_gene(n, g)
    return [round(values[n], 3) for n in WEIGHT_NAMES]


def clip(genome):
    out = []
    for n, g in zip(GENES, genome):
        lo, hi = BOUNDS[n]
        out.append(min(max(g, _to_gene(n, lo)), _to_gene(n, hi)))
    return out


# ---------------------------------------------------------------------------
# Operadores genéticos
# ---------------------------------------------------------------------------
def init_population(size, rng, sigma):
    """El individuo conocido (pesos por defecto) + mutaciones a su alrededor."""
    base = encode(DEFAULT_WEIGHTS)
    pop = [base]
    while len(pop) < size:
        pop.append(clip([g + rng.gauss(0, sigma * (2 if n == "CONNECT" else 1))
                         for n, g in zip(GENES, base)]))
    return pop


def tournament(pop, fitness, k, rng):
    """Selección por torneo: k individuos al azar, gana el de mayor fitness."""
    idx = rng.sample(range(len(pop)), k)
    return pop[max(idx, key=lambda i: fitness[i])]


def crossover(a, b, rng):
    """Cruce aritmético: hijo1 = α·a + (1−α)·b, hijo2 = α·b + (1−α)·a."""
    alpha = rng.random()
    c1 = [alpha * x + (1 - alpha) * y for x, y in zip(a, b)]
    c2 = [alpha * y + (1 - alpha) * x for x, y in zip(a, b)]
    return c1, c2


def mutate(genome, rng, p, sigma):
    """Mutación gaussiana gen a gen (en log: multiplica el peso por e^N(0,σ))."""
    return clip([g + rng.gauss(0, sigma) if rng.random() < p else g for g in genome])


def diversity(pop):
    """Desviación típica media de los genes (en log): 0 = población clonada."""
    n = len(pop)
    total = 0.0
    for j in range(len(GENES)):
        col = [ind[j] for ind in pop]
        m = sum(col) / n
        total += math.sqrt(sum((x - m) ** 2 for x in col) / n)
    return total / len(GENES)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------
TIME_LIMIT = None      # si se fija (--time), el fitness usa el motor real (prof. iterativa)


def _game(job):
    i, j, wb, ww, opening, depth, max_turns = job[:7]
    time_limit = job[7] if len(job) > 7 else None
    winner, turns = play_weights(wb, ww, opening, depth, max_turns, time_limit)
    return i, j, winner, turns


def round_robin(pool, weights, openings, depth, max_turns):
    """Liguilla: cada pareja (i, j) juega cada apertura con los dos colores.
    Devuelve (fitness medio por partida, nº de tablas, nº de partidas)."""
    jobs = []
    for i, j in combinations(range(len(weights)), 2):
        for op in openings:
            jobs.append((i, j, weights[i], weights[j], op, depth, max_turns, TIME_LIMIT))   # i negras
            jobs.append((j, i, weights[j], weights[i], op, depth, max_turns, TIME_LIMIT))   # i blancas
    n = len(weights)
    points = [0.0] * n
    games = [0] * n
    pair = [[0.0] * n for _ in range(n)]     # pair[i][j] = puntos de i contra j
    draws = 0
    for b, w, winner, _ in pool.imap_unordered(_game, jobs):
        games[b] += 1
        games[w] += 1
        if winner == Defines.BLACK:
            points[b] += 1
            pair[b][w] += 1
        elif winner == Defines.WHITE:
            points[w] += 1
            pair[w][b] += 1
        else:
            points[b] += 0.5
            points[w] += 0.5
            pair[b][w] += 0.5
            pair[w][b] += 0.5
            draws += 1
    round_robin.last_pair = pair
    return [p / g for p, g in zip(points, games)], draws, len(jobs)


def panel_fitness(pool, weights, panel, openings, depth, max_turns):
    """Fitness contra un PANEL FIJO de rivales (no contra la propia población):
    cada individuo juega cada apertura con los dos colores contra cada rival del
    panel. Así el fitness no depende de lo parecida que sea la población."""
    n = len(weights)
    jobs = []
    for i in range(n):
        for p, wp in enumerate(panel):
            for op in openings:
                jobs.append((i, n + p, weights[i], wp, op, depth, max_turns))
                jobs.append((n + p, i, wp, weights[i], op, depth, max_turns))
    points = [0.0] * n
    games = [0] * n
    draws = 0
    for b, w, winner, _ in pool.imap_unordered(_game, jobs):
        me = b if b < n else w
        games[me] += 1
        if winner == Defines.DRAW:
            points[me] += 0.5
            draws += 1
        elif (winner == Defines.BLACK) == (me == b):
            points[me] += 1
    return [p / g for p, g in zip(points, games)], draws, len(jobs)


def versus_default(pool, w, openings, depth, max_turns):
    """Puntos de w contra los pesos por defecto (fracción de 1)."""
    jobs = []
    for op in openings:
        jobs.append((0, 1, w, DEFAULT_WEIGHTS, op, depth, max_turns))
        jobs.append((1, 0, DEFAULT_WEIGHTS, w, op, depth, max_turns))
    pts = 0.0
    for b, wh, winner, _ in pool.imap_unordered(_game, jobs):
        if winner == Defines.DRAW:
            pts += 0.5
        elif (winner == Defines.BLACK and b == 0) or (winner == Defines.WHITE and wh == 0):
            pts += 1
    return pts / len(jobs)


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------
def run(pop_size=10, gens=15, k=3, p_cross=0.9, p_mut=0.2, sigma_mut=0.3,
        sigma_init=0.5, elite=2, openings_per_gen=4, val_openings=6, depth=2,
        max_turns=80, seed=0, workers=4, out=None, log=print,
        fitness_mode="roundrobin", hof_size=3):
    """fitness_mode: "roundrobin" (liguilla en la población, versión 1) o
    "panel" (contra los pesos por defecto + los hof_size últimos campeones
    de generación, "salón de la fama"; versión 2)."""
    rng = random.Random(seed)
    hall_of_fame = []
    anchor_fitness = None
    pop = init_population(pop_size, rng, sigma_init)
    val_ops = make_openings(val_openings, seed=999)       # fijas: miden el progreso
    history = []
    with Pool(workers) as pool:
        for gen in range(gens):
            t0 = time.perf_counter()
            weights = [decode(g) for g in pop]
            ops = make_openings(openings_per_gen, seed=1000 + gen + 100 * seed)
            if fitness_mode == "panel":
                panel = [list(DEFAULT_WEIGHTS)] + hall_of_fame[-hof_size:]
                fitness, draws, n_games = panel_fitness(pool, weights, panel, ops, depth, max_turns)
            elif fitness_mode == "anchor":
                # Liguilla con un "ancla": los pesos por defecto juegan siempre en
                # la liguilla (no se seleccionan ni se reproducen). Mide el progreso
                # dentro del propio fitness y frena la deriva hacia estrategias que
                # solo funcionan contra los hermanos.
                f_all, draws, n_games = round_robin(pool, weights + [list(DEFAULT_WEIGHTS)],
                                                    ops, depth, max_turns)
                fitness, anchor_fitness = f_all[:-1], f_all[-1]
            else:
                fitness, draws, n_games = round_robin(pool, weights, ops, depth, max_turns)
            order = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)
            best = order[0]
            if weights[best] not in hall_of_fame and weights[best] != list(DEFAULT_WEIGHTS):
                hall_of_fame.append(weights[best])
            if val_openings:
                vs_def = versus_default(pool, weights[best], val_ops, depth, max_turns)
            else:   # (modo ancla) resultado directo del campeón contra el ancla en la liguilla
                vs_def = round_robin.last_pair[best][-1] / (2 * openings_per_gen)
            rec = {"gen": gen, "best_fitness": fitness[best],
                   "mean_fitness": sum(fitness) / len(fitness),
                   "best_weights": weights[best], "best_vs_default": vs_def,
                   "default_in_pop_fitness": fitness[0] if gen == 0 else None,
                   "anchor_fitness": anchor_fitness if fitness_mode == "anchor" else None,
                   "diversity": diversity(pop), "draw_rate": draws / n_games,
                   "fitness": fitness, "population": weights,
                   "seconds": time.perf_counter() - t0}
            history.append(rec)
            log(f"gen {gen:2d}  best {fitness[best]:.3f}  media {rec['mean_fitness']:.3f}  "
                f"vs_default {vs_def:.3f}  ancla {anchor_fitness if anchor_fitness is None else round(anchor_fitness, 3)}  div {rec['diversity']:.3f}  "
                f"tablas {rec['draw_rate']:.2f}  {rec['seconds']:.0f}s  {weights[best]}")
            if out:
                with open(out, "w") as f:
                    json.dump({"params": dict(pop_size=pop_size, gens=gens, k=k, p_cross=p_cross,
                                              p_mut=p_mut, sigma_mut=sigma_mut, sigma_init=sigma_init,
                                              elite=elite, openings_per_gen=openings_per_gen,
                                              fitness_mode=fitness_mode, hof_size=hof_size,
                                              depth=depth, max_turns=max_turns, seed=seed),
                               "history": history}, f, indent=1)
            if gen == gens - 1:
                break
            # Reemplazo: élite + hijos (selección por torneo, cruce, mutación)
            new_pop = [pop[i] for i in order[:elite]]
            while len(new_pop) < pop_size:
                a = tournament(pop, fitness, k, rng)
                b = tournament(pop, fitness, k, rng)
                if rng.random() < p_cross:
                    c1, c2 = crossover(a, b, rng)
                else:
                    c1, c2 = list(a), list(b)
                new_pop.append(mutate(c1, rng, p_mut, sigma_mut))
                if len(new_pop) < pop_size:
                    new_pop.append(mutate(c2, rng, p_mut, sigma_mut))
            pop = new_pop
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pmut", type=float, default=0.2)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--openings", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=int, default=6)
    ap.add_argument("--fitness", choices=["roundrobin", "panel", "anchor"], default="roundrobin")
    ap.add_argument("--time", type=float, default=None,
                    help="segundos por jugada: fitness con el motor real en vez de prof. fija")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="ga_run.json")
    a = ap.parse_args()
    TIME_LIMIT = a.time
    run(a.pop, a.gens, a.k, 0.9, a.pmut, a.sigma, 0.5, a.elite, a.openings, a.val,
        a.depth, a.max_turns, a.seed, a.workers, a.out, fitness_mode=a.fitness)
