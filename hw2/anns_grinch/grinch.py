# Solution by team:
# - Marcin Wojnarowski (376886)
# - Jonathan Arnoult (369910)
# - Emilien Ganier (369941)

import math
import sys
from random import shuffle


def hamming_dist(x: list[int], y: list[int], dim: int):
    count = 0
    for i in range(dim):
        if x[i] != y[i]:
            count += 1
    return count

def print_vector(x: list[int]):
    print(" ".join(map(str, x)))
    sys.stdout.flush()

def report_solution(x: list[int]):
    print("* ", end="")
    sys.stdout.flush()
    print_vector(x)

class OnlineANNS:
    def __init__(self, dim: int):
        self.d = dim

    def query(self, q: list[int]) -> list[int] | None:
        print("q ", end="")
        sys.stdout.flush()
        print_vector(q)

        line = input().split(' ')

        size = int(line[0])
        if size == 1:
            return None
        else:
            return [1, 2]

def main():
    header = input().split(' ')
    d, r, c, n, N = map(int, header)

    line = input().split(' ')
    z = list(map(int, line))

    ds = OnlineANNS(d)

    mu = min(r, math.ceil(2.0 * math.e * math.e * (math.log(n) + 1.0)))

    for _ in range(N):
        indices = list(range(d))
        shuffle(indices)

        q = z.copy()
        for i in range(r - mu):
            q[indices[i]] = 1 - q[indices[i]]

        is_not_bottom = ds.query(q) is not None

        while is_not_bottom and hamming_dist(q, z, d) < r:
            w = math.ceil(c * r) + 1 - hamming_dist(q, z, d)

            I = [i for i in range(d) if q[i] == z[i]]
            shuffle(I)
            I = I[:w]

            u = [q.copy()]
            for i in range(w):
                previous = u[-1].copy()
                previous[I[i]] = 1 - previous[I[i]]
                u.append(previous)
            
            for i in range(1, len(u)):
                if ds.query(u[i]) is None:
                    q[I[i - 1]] = 1 - q[I[i - 1]]
                    is_not_bottom = ds.query(q) is not None
                    break


        if not is_not_bottom and hamming_dist(q, z, d) <= r:
            report_solution(q)
            return 0

    return 1

if __name__ == "__main__":
    sys.exit(main())
