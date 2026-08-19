"""Game engine implemented strictly from uttt-rules-spec.md. Stdlib only."""

# The 8 winning lines of a 3x3 grid, as (row, col) triples.
_LINES = (
    [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
    [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)],
)

_DRAWN = "D"  # small-board decided with no owner; master cell stays unmarked


class Engine:
    def __init__(self, level: int):
        if level not in (1, 2):
            raise ValueError("level must be 1 or 2")
        self.level = level
        n = 3 if level == 1 else 9
        self._cells = [[None] * n for _ in range(n)]  # None / 0 / 1
        self._turn = 0
        self._over = False
        self._winner = -1
        self._scores = [0, 0]
        if level == 2:
            self._owner = [[None] * 3 for _ in range(3)]  # None / 0 / 1 / _DRAWN
            self._last = None  # previous move (row, col), or None

    # -- helpers ------------------------------------------------------------

    def _has_line(self, player, r0=0, c0=0):
        """3-in-a-row for player in the 3x3 grid whose top-left cell is (r0, c0)."""
        c = self._cells
        return any(all(c[r0 + r][c0 + q] == player for r, q in line) for line in _LINES)

    def _board_empties(self, br, bc):
        return {
            (br * 3 + r, bc * 3 + q)
            for r in range(3)
            for q in range(3)
            if self._cells[br * 3 + r][bc * 3 + q] is None
        }

    def _master_line(self, player):
        o = self._owner
        return any(all(o[r][q] == player for r, q in line) for line in _LINES)

    # -- interface ----------------------------------------------------------

    def valid_actions(self):
        if self._over:
            return set()
        if self.level == 1:
            return {
                (r, c) for r in range(3) for c in range(3)
                if self._cells[r][c] is None
            }
        if self._last is not None:
            tb = (self._last[0] % 3, self._last[1] % 3)
            if self._owner[tb[0]][tb[1]] is None:  # not decided
                return self._board_empties(*tb)
        # first move, or target board decided: all empties of not-owned boards
        acts = set()
        for br in range(3):
            for bc in range(3):
                if self._owner[br][bc] not in (0, 1):
                    acts |= self._board_empties(br, bc)
        return acts

    def play(self, row: int, col: int) -> None:
        if (row, col) not in self.valid_actions():
            raise ValueError(f"invalid move ({row}, {col})")
        p = self._turn % 2
        self._cells[row][col] = p
        if self.level == 1:
            if self._has_line(p):
                self._scores[p] = 10
                self._over, self._winner = True, p
            elif all(v is not None for r in self._cells for v in r):
                self._over = True  # 0-0 draw
        else:
            br, bc = row // 3, col // 3
            if self._has_line(p, br * 3, bc * 3):
                self._owner[br][bc] = p
                self._scores[p] += 1
                if self._master_line(p):
                    self._scores[p] = 10
                    self._over, self._winner = True, p
            elif not self._board_empties(br, bc):
                self._owner[br][bc] = _DRAWN
            self._last = (row, col)
            if not self._over and not self.valid_actions():
                self._over = True
                s0, s1 = self._scores
                self._winner = 0 if s0 > s1 else 1 if s1 > s0 else -1
        self._turn += 1

    @property
    def current_player(self) -> int:
        return self._turn % 2

    @property
    def game_over(self) -> bool:
        return self._over

    @property
    def scores(self):
        return tuple(self._scores)

    @property
    def winner(self) -> int:
        return self._winner


if __name__ == "__main__":
    import random

    for level in (1, 2):
        for seed in range(20):
            rng = random.Random(1000 * level + seed)
            e = Engine(level)
            t = 0
            while not e.game_over:
                assert e.current_player == t % 2, "alternation broken"
                acts = e.valid_actions()
                assert acts, "no actions but game not over"
                e.play(*rng.choice(sorted(acts)))
                t += 1
            assert e.valid_actions() == set(), "actions after game over"
            s0, s1 = e.scores
            assert 0 <= s0 <= 10 and 0 <= s1 <= 10, "score out of bounds"
            if level == 1:
                assert (s0, s1) in {(0, 0), (10, 0), (0, 10)}
            if e.winner == 0:
                assert s0 > s1
            elif e.winner == 1:
                assert s1 > s0
            else:
                assert s0 == s1
            try:
                e.play(0, 0)
            except ValueError:
                pass
            else:
                raise AssertionError("play after game over did not raise")

    # directed check: level 1 fastest win
    e = Engine(1)
    for mv in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]:
        e.play(*mv)
    assert e.game_over and e.winner == 0 and e.scores == (10, 0)

    # directed check: level 2 target-board constraint
    e = Engine(2)
    e.play(4, 4)  # local (1,1) -> target board (1,1), the center board
    assert e.valid_actions() == {
        (r, c) for r in range(3, 6) for c in range(3, 6) if (r, c) != (4, 4)
    }

    print("self-check passed")
