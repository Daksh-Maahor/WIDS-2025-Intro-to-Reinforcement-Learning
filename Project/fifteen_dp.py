"""15‑puzzle solver using simple dynamic programming + A*.

We roughly follow Amin Shali’s idea of solving the puzzle row‑by‑row with
masked state spaces, then fall back to A* if the learned policy stalls.
"""

import random
import heapq
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable


BOARD_SIZE = 4
NUM_TILES = BOARD_SIZE * BOARD_SIZE
EMPTY = NUM_TILES  # use 16 as empty


Position = Tuple[int, ...]  # length-16 tuple of ints in 0..16, where EMPTY=16
Action = int  # we use this both for "tile numbers" and "board indices" depending on context


def encode_state(pos: Position) -> int:
    """Pack a 16-cell board into a single integer (5 bits per cell)."""
    code = 0
    for t in pos:
        code = (code << 5) | (t & 0x1F)
    return code


def decode_state(code: int) -> Position:
    """Unpack an integer created by encode_state back into a board."""
    cells = [0] * NUM_TILES
    for i in range(NUM_TILES - 1, -1, -1):
        cells[i] = code & 0x1F
        code >>= 5
    return tuple(cells)  # type: ignore[return-value]


def index_to_rc(idx: int) -> Tuple[int, int]:
    return divmod(idx, BOARD_SIZE)


def rc_to_index(r: int, c: int) -> int:
    return r * BOARD_SIZE + c


def manhattan_distance(pos: Position) -> int:
    """Heuristic: sum of manhattan distances of tiles from their goal positions."""
    dist = 0
    for idx, tile in enumerate(pos):
        if tile == EMPTY:
            continue
        goal_idx = tile - 1
        r1, c1 = index_to_rc(idx)
        r2, c2 = index_to_rc(goal_idx)
        dist += abs(r1 - r2) + abs(c1 - c2)
    return dist


def is_solved(pos: Position) -> bool:
    return pos == tuple(list(range(1, NUM_TILES)) + [EMPTY])


def legal_moves(pos: Position) -> List[Action]:
    """Return list of tile numbers that can move into the empty cell."""
    empty_idx = pos.index(EMPTY)
    er, ec = index_to_rc(empty_idx)
    moves: List[Action] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = er + dr, ec + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            tile_idx = rc_to_index(nr, nc)
            moves.append(pos[tile_idx])
    return moves


def legal_move_indices(pos: Position) -> List[int]:
    """Return list of *indices* whose tiles can move into the empty cell."""
    empty_idx = pos.index(EMPTY)
    er, ec = index_to_rc(empty_idx)
    indices: List[int] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = er + dr, ec + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            idx = rc_to_index(nr, nc)
            indices.append(idx)
    return indices


def apply_move(pos: Position, tile: Action) -> Position:
    """Return new position after sliding `tile` into the empty cell."""
    empty_idx = pos.index(EMPTY)
    tile_idx = pos.index(tile)
    new_pos = list(pos)
    new_pos[empty_idx], new_pos[tile_idx] = new_pos[tile_idx], new_pos[empty_idx]
    return tuple(new_pos)  # type: ignore[return-value]


def apply_move_index(pos: Position, tile_index: int) -> Position:
    """Return new position after sliding the tile at `tile_index` into the empty cell."""
    empty_idx = pos.index(EMPTY)
    new_pos = list(pos)
    new_pos[empty_idx], new_pos[tile_index] = new_pos[tile_index], new_pos[empty_idx]
    return tuple(new_pos)  # type: ignore[return-value]


def mask_position(pos: Position, keep_tiles: Iterable[int]) -> Position:
    """Zero‑out tiles that are not in keep_tiles (except EMPTY)."""
    keep = set(keep_tiles)
    masked = []
    for t in pos:
        if t == EMPTY:
            masked.append(EMPTY)
        elif t in keep:
            masked.append(t)
        else:
            masked.append(0)
    return tuple(masked)  # type: ignore[return-value]


def astar_solve(start: Position, max_expansions: int = 500_000) -> List[Action]:
    """Fallback exact solver using A* with Manhattan distance heuristic.

    This is NOT part of the original DP idea, but it guarantees that if the
    board is solvable and we have enough search budget, we will find a complete
    solution even if the learned DP policy is approximate.
    """
    start_code = encode_state(start)
    goal = tuple(list(range(1, NUM_TILES)) + [EMPTY])  # type: ignore[assignment]
    goal_code = encode_state(goal)

    # priority queue of (f=g+h, g, state_code, Position)
    heap: List[Tuple[int, int, int, Position]] = []
    heapq.heappush(heap, (manhattan_distance(start), 0, start_code, start))

    # best known g-value for each visited state
    g_best: Dict[int, int] = {start_code: 0}
    # parent pointers for reconstruction: state_code -> (parent_code, tile_moved)
    parent: Dict[int, Tuple[int, Action]] = {}

    expansions = 0

    while heap and expansions < max_expansions:
        f, g, code, pos = heapq.heappop(heap)
        if code == goal_code:
            # reconstruct path of tile moves
            moves: List[Action] = []
            cur = code
            while cur != start_code:
                p_code, move = parent[cur]
                moves.append(move)
                cur = p_code
            moves.reverse()
            return moves

        if g != g_best.get(code, float("inf")):
            continue  # outdated entry

        expansions += 1

        for tile in legal_moves(pos):
            nxt = apply_move(pos, tile)
            nxt_code = encode_state(nxt)
            g2 = g + 1
            if g2 >= g_best.get(nxt_code, float("inf")):
                continue
            g_best[nxt_code] = g2
            parent[nxt_code] = (code, tile)
            h2 = manhattan_distance(nxt)
            heapq.heappush(heap, (g2 + h2, g2, nxt_code, nxt))

    # failed to find within budget
    return []


@dataclass(frozen=True)
class StageSpec:
    name: str
    keep_tiles: Tuple[int, ...]
    # terminal-test is applied on the *unmasked* position
    def terminal(self, pos: Position) -> bool:
        if self.name == "row1":
            # 1,2,3,4 in the first row
            goal = (1, 2, 3, 4)
            return pos[0:4] == goal
        elif self.name == "row2":
            # first two rows solved
            goal = (1, 2, 3, 4, 5, 6, 7, 8)
            return pos[0:8] == goal
        elif self.name == "last":
            return is_solved(pos)
        else:
            raise ValueError(f"Unknown stage {self.name}")


ROW1_STAGE = StageSpec(name="row1", keep_tiles=(1, 2, 3, 4, EMPTY))
ROW2_STAGE = StageSpec(name="row2", keep_tiles=(5, 6, 7, 8, EMPTY))
LAST_STAGE = StageSpec(
    name="last",
    keep_tiles=(9, 10, 11, 12, 13, 14, 15, EMPTY),
)


class ValueIterationSolver:
    """Small value‑iteration based policy for the 15‑puzzle."""

    def __init__(
        self,
        gamma: float = 0.95,
        theta: float = 1e-6,
        move_cost: float = -1.0,
    ) -> None:
        self.gamma = gamma
        self.theta = theta
        self.move_cost = move_cost

        # Hard cap on number of states per stage. Increase if you want.
        self.max_states_per_stage: int = 50_000

        # value functions and policies, per‑stage, keyed by compact int state.
        self.V: Dict[str, Dict[int, float]] = {}
        self.pi: Dict[str, Dict[int, Action]] = {}

    # ----- hashing / canonical representation -----

    @staticmethod
    def hash_state(pos: Position) -> int:
        """Return compact integer representation used as hash key."""
        return encode_state(pos)

    # ----- state generation -----

    def _generate_reachable_states(
        self,
        stage: StageSpec,
    ) -> Dict[int, None]:
        """Backward BFS from terminal states for a given stage."""
        seen: Dict[int, None] = {}
        q: deque[Position] = deque()

        # enumerate all full terminal states for this stage
        terminal_full_states = enumerate_terminal_states(stage)
        for full in terminal_full_states:
            masked = mask_position(full, stage.keep_tiles)
            h = self.hash_state(masked)
            if h not in seen:
                seen[h] = None
                q.append(full)
                if len(seen) >= self.max_states_per_stage:
                    # memory-safe cap on number of stored states
                    return seen

        while q and len(seen) < self.max_states_per_stage:
            full_pos = q.popleft()
            masked = mask_position(full_pos, stage.keep_tiles)
            h = self.hash_state(masked)

            # explore predecessors by trying all legal moves backwards
            for tile in legal_moves(full_pos):
                prev = apply_move(full_pos, tile)  # moving tile into empty is symmetric
                prev_masked = mask_position(prev, stage.keep_tiles)
                h_prev = self.hash_state(prev_masked)
                if h_prev not in seen:
                    seen[h_prev] = None
                    if len(seen) >= self.max_states_per_stage:
                        break
                    q.append(prev)

        return seen

    # ----- DP core -----

    def _value_iteration_stage(self, stage: StageSpec) -> None:
        """Run value iteration for a single stage and store V, pi."""
        # generate state space (masked)
        reachable = self._generate_reachable_states(stage)
        V: Dict[int, float] = {s: 0.0 for s in reachable.keys()}

        terminal_masked: set[int] = set()
        for full in enumerate_terminal_states(stage):
            m = mask_position(full, stage.keep_tiles)
            terminal_masked.add(self.hash_state(m))

        while True:
            delta = 0.0
            for state_code in list(V.keys()):
                if state_code in terminal_masked:
                    continue  # V(t) = 0, fixed

                masked_state = decode_state(state_code)

                full_like = masked_state

                best = float("-inf")
                for idx in legal_move_indices(full_like):
                    sp_full = apply_move_index(full_like, idx)
                    sp_masked = mask_position(sp_full, stage.keep_tiles)
                    sp_h = self.hash_state(sp_masked)
                    if sp_h not in V:
                        # should not normally happen, but guard anyway
                        continue
                    r = self.move_cost  # constant step penalty
                    val = r + self.gamma * V[sp_h]
                    if val > best:
                        best = val

                if best == float("-inf"):
                    # dead-end (shouldn't occur in a proper state graph)
                    continue

                v_old = V[state_code]
                V[state_code] = best
                delta = max(delta, abs(v_old - best))

            if delta < self.theta:
                break

        # policy extraction
        pi: Dict[int, Action] = {}
        for state_code in V.keys():
            if state_code in terminal_masked:
                continue
            full_like = decode_state(state_code)

            best_val = float("-inf")
            best_action: Action | None = None
            for idx in legal_move_indices(full_like):
                sp_full = apply_move_index(full_like, idx)
                sp_masked = mask_position(sp_full, stage.keep_tiles)
                sp_h = self.hash_state(sp_masked)
                if sp_h not in V:
                    continue
                r = self.move_cost
                val = r + self.gamma * V[sp_h]
                if val > best_val:
                    best_val = val
                    best_action = idx
            if best_action is not None:
                pi[state_code] = best_action

        self.V[stage.name] = V
        self.pi[stage.name] = pi

    # ----- public API -----

    def train_all_stages(self) -> None:
        """Compute value functions and policies for all three stages."""
        for stage in (ROW1_STAGE, ROW2_STAGE, LAST_STAGE):
            print(f"Training stage: {stage.name}")
            self._value_iteration_stage(stage)

    def _choose_action(self, stage: StageSpec, pos: Position) -> Action | None:
        masked = mask_position(pos, stage.keep_tiles)
        h = self.hash_state(masked)
        policy = self.pi.get(stage.name, {})
        return policy.get(h, None)

    def solve(self, start: Position, max_steps: int = 500) -> List[Action]:
        """Solve a puzzle by running greedy policy across the three stages.

        Returns the sequence of tile moves. If solution not found within
        `max_steps` moves, the sequence may be partial.
        """
        pos = start
        moves: List[Action] = []

        stages = [ROW1_STAGE, ROW2_STAGE, LAST_STAGE]
        stage_idx = 0
        current_stage = stages[stage_idx]

        for _ in range(max_steps):
            if current_stage.terminal(pos):
                stage_idx += 1
                if stage_idx >= len(stages):
                    break
                current_stage = stages[stage_idx]

            a = self._choose_action(current_stage, pos)
            if a is None:
                break  # no policy action known
            # Record the actual tile number being moved for readability,
            # but use the index-based action for the state transition.
            tile_moved = pos[a]
            pos = apply_move_index(pos, a)
            moves.append(tile_moved)

            if is_solved(pos):
                break

        # If DP policy did not fully solve the board, fall back to A* search
        if not is_solved(pos):
            extra = astar_solve(pos)
            moves.extend(extra)

        return moves


def enumerate_terminal_states(stage: StageSpec) -> List[Position]:
    """Enumerate *sampled* terminal full-board states for a stage.

    In theory, for a stage like "row1" we could take *all* permutations of
    the remaining 12 cells. That is astronomically many states and is not
    practical to enumerate.

    Instead, we:
        - fix the portion of the board that must be solved for this stage,
        - randomise the rest of the board a number of times,
        - treat those as a *sample* of terminal states from which we will
          run a backward BFS.

    This keeps the code small, fast, and close in spirit to the Java version,
    but you should keep in mind that the resulting policy is approximate.
    """
    # base solved board
    goal = tuple(list(range(1, NUM_TILES)) + [EMPTY])

    if stage.name == "row1":
        # First row fixed as in goal; everything else arbitrary.
        fixed_indices = list(range(4))
    elif stage.name == "row2":
        fixed_indices = list(range(8))
    elif stage.name == "last":
        return [goal]
    else:
        raise ValueError(f"Unknown stage {stage.name}")

    fixed_values = {i: goal[i] for i in fixed_indices}

    remaining_indices = [i for i in range(NUM_TILES) if i not in fixed_indices]
    remaining_values = [goal[i] for i in remaining_indices]

    # For simplicity (and to keep runtime/sizes reasonable), instead of
    # enumerating *all* permutations of the remaining cells, we sample
    # a subset. Increase num_samples for a richer (but slower) policy.
    terminals: List[Position] = []
    num_samples = 50  # keep tiny; increase if you want more thorough policies

    for _ in range(num_samples):
        # Work on a shallow copy so shuffling does not bias other samples.
        shuffled = list(remaining_values)
        random.shuffle(shuffled)
        board = [0] * NUM_TILES
        for i in fixed_indices:
            board[i] = fixed_values[i]
        for idx, cell in enumerate(remaining_indices):
            board[cell] = shuffled[idx]
        terminals.append(tuple(board))  # type: ignore[arg-type]

    # include the clean goal for good measure
    if goal not in terminals:
        terminals.append(goal)
    return terminals


def scramble(start: Position | None = None, steps: int = 50) -> Position:
    """Generate a scrambled puzzle by doing random legal moves."""
    if start is None:
        start = tuple(list(range(1, NUM_TILES)) + [EMPTY])  # type: ignore[assignment]
    pos = start
    for _ in range(steps):
        mv = random.choice(legal_moves(pos))
        pos = apply_move(pos, mv)
    return pos


def pretty_print(pos: Position) -> None:
    """Print board in 4x4 format."""
    print("-" * 21)
    for r in range(BOARD_SIZE):
        row = []
        for c in range(BOARD_SIZE):
            t = pos[rc_to_index(r, c)]
            if t == EMPTY:
                row.append("  ")
            else:
                row.append(f"{t:2d}")
        print("| " + " ".join(row) + " |")
    print("-" * 21)


def main() -> None:
    solved = tuple(list(range(1, NUM_TILES)) + [EMPTY])  # type: ignore[assignment]
    # Start from a randomly scrambled (and therefore solvable) board.
    # Each run will use a fresh random scramble.
    start = scramble(solved, steps=100)

    print("Scrambled start:")
    pretty_print(start)

    solver = ValueIterationSolver(gamma=0.95, theta=1e-3, move_cost=-1.0)
    solver.train_all_stages()

    moves = solver.solve(start, max_steps=200)
    print(f"\nPlanned {len(moves)} moves (tile numbers):")
    print(moves)

    # Replay the entire solution as a sequence of boards.
    pos = start
    print("\nSolution trace:")
    pretty_print(pos)
    for step, mv in enumerate(moves, start=1):
        pos = apply_move(pos, mv)
        print(f"\nStep {step}: move tile {mv}")
        pretty_print(pos)

    print(f"Solved: {is_solved(pos)}")


if __name__ == "__main__":
    main()

