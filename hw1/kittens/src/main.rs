#![allow(clippy::needless_range_loop)]

/// Solution by team:
/// - Marcin Wojnarowski (376886)
/// - Jonathan Arnoult (369910)
/// - Emilien Ganier (369941)
use std::{
    cmp::max,
    collections::HashMap,
    fmt::Debug,
    io::{self, BufRead},
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

/// Rust std has no random generators. This is based on:
/// https://github.com/rust-lang/rust/blob/1.55.0/library/core/src/slice/sort.rs#L559-L573
fn random_numbers() -> impl FnMut() -> u32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let mut random = RandomState::new().build_hasher().finish() as u32;
    move || {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        random
    }
}

/// A prime number used for Zp arithmetic. Number chosen to be big enough to handle the problem.
const FIELD_ORDER: usize = 20011;

/// Represents an element of the field Z/pZ for a prime number `P`.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct Zp<const P: usize>(usize);

impl<const P: usize> Debug for Zp<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Zp<{}>({})", P, self.0)
    }
}

impl<const P: usize> Add<Self> for Zp<P> {
    type Output = Self;

    fn add(self, rhs: Zp<P>) -> Self::Output {
        Self((self.0 + rhs.0) % P)
    }
}

impl<const P: usize> Mul<Self> for Zp<P> {
    type Output = Self;

    fn mul(self, rhs: Zp<P>) -> Self::Output {
        Self((self.0 * rhs.0) % P)
    }
}

impl<const P: usize> Sub<Self> for Zp<P> {
    type Output = Self;

    fn sub(self, rhs: Zp<P>) -> Self::Output {
        Self((self.0 + P - rhs.0) % P)
    }
}

impl<const P: usize> Div<Self> for Zp<P> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Zp<P>) -> Self::Output {
        self * rhs.mutliplicative_inverse()
    }
}

impl<const P: usize> AddAssign<Self> for Zp<P> {
    fn add_assign(&mut self, rhs: Zp<P>) {
        *self = *self + rhs;
    }
}

impl<const P: usize> MulAssign<Self> for Zp<P> {
    fn mul_assign(&mut self, rhs: Zp<P>) {
        *self = *self * rhs;
    }
}

impl<const P: usize> SubAssign<Self> for Zp<P> {
    fn sub_assign(&mut self, rhs: Zp<P>) {
        *self = *self - rhs;
    }
}

impl<const P: usize> Zp<P> {
    const ZERO: Self = Self::new(0);

    const fn new(value: usize) -> Self {
        Self(value % P)
    }

    /// Finds a `b` such that `a * b = 1 (mod P)`.
    /// Fails for `a = 0`.
    fn mutliplicative_inverse(&self) -> Self {
        if self == &Self::ZERO {
            panic!("Multiplicative inverse of zero is undefined");
        }

        // Extended Euclidean algorithm
        let (mut old_r, mut r) = (self.0 as i64, P as i64);
        let (mut old_s, mut s) = (1, 0);
        let (mut old_t, mut t) = (0, 1);

        while r != 0 {
            let quotient = old_r / r;
            (old_r, r) = (r, old_r - quotient * r);
            (old_s, s) = (s, old_s - quotient * s);
            (old_t, t) = (t, old_t - quotient * t);
        }

        Self::new(if old_s < 0 { P as i64 + old_s } else { old_s } as _)
    }

    fn pow(&self, mut exp: usize) -> Self {
        let mut res = Self::new(1);
        let mut base = *self;

        while exp > 0 {
            if exp % 2 == 1 {
                res *= base;
            }

            base *= base;
            exp /= 2;
        }

        res
    }
}

/// Represents a square matrix over a Zp field.
/// Row-major indexing.
#[derive(Debug, PartialEq, Clone)]
struct Matrix<const P: usize> {
    data: Vec<Zp<P>>,
    // Invariant: size * size == data.len()
    size: usize,
}

impl<const P: usize> Mul<Matrix<P>> for Matrix<P> {
    type Output = Matrix<P>;

    fn mul(self, rhs: Matrix<P>) -> Self::Output {
        assert_eq!(self.size(), rhs.size());

        let n = self.size();
        let mut res = Self::new(n);

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    res[(i, j)] += self[(i, k)] * rhs[(k, j)];
                }
            }
        }

        res
    }
}

impl<const P: usize> Mul<Vec<Zp<P>>> for Matrix<P> {
    type Output = Vec<Zp<P>>;

    fn mul(self, rhs: Vec<Zp<P>>) -> Self::Output {
        assert_eq!(self.size(), rhs.len());

        let n = self.size();
        let mut res = vec![Zp::ZERO; n];

        for i in 0..n {
            for j in 0..n {
                res[i] += self[(i, j)] * rhs[j];
            }
        }

        res
    }
}

impl<const P: usize, const N: usize> From<[[usize; N]; N]> for Matrix<P> {
    fn from(data: [[usize; N]; N]) -> Self {
        let mut res = Self::new(N);

        for i in 0..N {
            for j in 0..N {
                res[(i, j)] = Zp::new(data[i][j]);
            }
        }

        res
    }
}

impl<const P: usize> Index<(usize, usize)> for Matrix<P> {
    type Output = Zp<P>;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.data[i * self.size() + j]
    }
}

impl<const P: usize> IndexMut<(usize, usize)> for Matrix<P> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        let n = self.size();
        &mut self.data[i * n + j]
    }
}

impl<const P: usize> Matrix<P> {
    fn new(size: usize) -> Self {
        let data = vec![Zp::ZERO; size * size];

        Self { data, size }
    }

    fn new_uniform_random(size: usize) -> Self {
        let mut rand = random_numbers();
        let data = (0..size * size).map(|_| Zp::new(rand() as _)).collect();

        Self { data, size }
    }

    fn new_identity(size: usize) -> Self {
        let mut res = Self::new(size);

        for i in 0..size {
            res[(i, i)] = Zp::new(1);
        }

        res
    }

    const fn size(&self) -> usize {
        self.size
    }

    fn swap_rows(&mut self, i: usize, j: usize) {
        self.data.swap(i, j);
    }

    /// No decomposition if the matrix is not invertible.
    fn lup_decompose(&self) -> Option<(Self, Self, usize)> {
        let n = self.size();
        let mut p = Self::new_identity(n);
        let mut lu = self.clone();
        let mut swaps = 0;

        for i in 0..n - 1 {
            let mut max_pivot = i;
            for k in i + 1..n {
                if lu[(k, i)] > lu[(i, i)] {
                    max_pivot = k;
                }
            }

            if lu[(max_pivot, i)] == Zp::ZERO {
                return None;
            }

            if max_pivot != i {
                swaps += 1;
                p.swap_rows(max_pivot, i);

                for k in i..n {
                    (lu[(i, k)], lu[(max_pivot, k)]) = (lu[(max_pivot, k)], lu[(i, k)]);
                }
            }

            for k in i + 1..n {
                lu[(k, i)] = lu[(k, i)] / lu[(i, i)];

                for j in i + 1..n {
                    lu[(k, j)] = lu[(k, j)] - lu[(i, j)] * lu[(k, i)];
                }
            }
        }

        Some((lu, p, swaps))
    }

    fn det(&self) -> Zp<P> {
        let Some((lu, _, swaps)) = self.lup_decompose() else {
            return Zp::ZERO;
        };

        let mut det = Zp::new(1);
        for i in 0..self.size() {
            det *= lu[(i, i)];
        }
        det *= if swaps % 2 == 0 {
            Zp::new(1)
        } else {
            Zp::new(P - 1)
        };

        det
    }
}

/// Represents a connection preference of a volunteer to a city and its cost.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Connection {
    volunteer: usize,
    city: usize,
    cost: usize,
}

/// Represents the problem input.
#[derive(Debug, PartialEq)]
struct Input {
    n: usize,
    budget: usize,
    train_cost: usize,
    car_cost: usize,
    connections: Vec<Connection>,
}

impl FromIterator<String> for Input {
    /// Given a list of stdin lines, parse them into an `Input` struct.
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        let mut iter = iter.into_iter();

        let header = iter.next().unwrap();
        let mut header = header.split(' ');
        let (n, m, b, t, c) = (
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
        );

        let connections = iter
            .map(|line| {
                let mut line = line.split(' ');
                let (i, j, w) = (
                    line.next().unwrap().parse().unwrap(),
                    line.next().unwrap().parse().unwrap(),
                    line.next().unwrap().parse().unwrap(),
                );

                assert!(w == t || w == c);
                assert!(i < n && j < n);

                Connection {
                    volunteer: i,
                    city: j,
                    cost: w,
                }
            })
            .collect::<Vec<_>>();

        assert_eq!(connections.len(), m);

        Self {
            n,
            budget: b,
            train_cost: t,
            car_cost: c,
            connections,
        }
    }
}

/// The amount of train and car rides to be taken to exactly exhaust the budget.
#[derive(Debug, PartialEq, Clone, Copy)]
struct CountCombination {
    train_count: usize,
    car_count: usize,
}

impl CountCombination {
    /// Returns `None` if there exists no such combination.
    const fn new(input: &Input) -> Option<Self> {
        // Solve two equations for integer solutions:
        // n = train_count + car_count
        // budget = train_count * train_cost + car_count * car_cost
        let (budget, n, train_cost, car_cost) = (
            input.budget as i64,
            input.n as i64,
            input.train_cost as i64,
            input.car_cost as i64,
        );
        let car_count = (budget - n * train_cost) / (car_cost - train_cost);
        let train_count = n - car_count;

        // Verify we got non-negative integer solutions by plugging it back to the equations.
        if n == train_count + car_count
            && budget == train_count * train_cost + car_count * car_cost
            && train_count >= 0
            && car_count >= 0
        {
            Some(CountCombination {
                train_count: train_count as _,
                car_count: car_count as _,
            })
        } else {
            None
        }
    }
}

struct LinearEquationSystem<const P: usize> {
    matrix: Matrix<P>,
    rhs: Vec<Zp<P>>,
}

impl<const P: usize> LinearEquationSystem<P> {
    fn new(matrix: Matrix<P>, rhs: Vec<Zp<P>>) -> Self {
        assert_eq!(matrix.size(), rhs.len());
        Self { matrix, rhs }
    }

    fn solve(self) -> Vec<Zp<P>> {
        let Self {
            mut matrix,
            mut rhs,
        } = self;
        let n = matrix.size();

        for i in 0..n - 1 {
            for j in i..n - 1 {
                reduce(&mut matrix, &mut rhs, i, j);
            }
        }

        for i in (1..n).rev() {
            eliminate(&mut matrix, &mut rhs, i);
        }

        let mut result = vec![Zp::ZERO; n];
        for i in 0..n {
            result[i] = rhs[i] / matrix[(i, i)];
        }

        return result;

        fn reduce<const P: usize>(matrix: &mut Matrix<P>, rhs: &mut [Zp<P>], i: usize, j: usize) {
            if matrix[(i, i)] == Zp::ZERO {
                return;
            }

            let factor = matrix[(j + 1, i)] / matrix[(i, i)];
            for k in i..matrix.size() {
                matrix[(j + 1, k)] = matrix[(j + 1, k)] - factor * matrix[(i, k)];
            }
            rhs[j + 1] -= factor * rhs[i];
        }

        fn eliminate<const P: usize>(matrix: &mut Matrix<P>, rhs: &mut [Zp<P>], i: usize) {
            if matrix[(i, i)] == Zp::ZERO {
                return;
            }

            for j in (1..=i).rev() {
                let factor = matrix[(j - 1, i)] / matrix[(i, i)];
                rhs[j - 1] -= factor * rhs[i];
                for k in (0..matrix.size()).rev() {
                    matrix[(j - 1, k)] = matrix[(j - 1, k)] - factor * matrix[(i, k)];
                }
            }
        }
    }
}

impl Input {
    fn solve(&self) -> bool {
        // run k times so that chance of failure is 0.1%
        let k = 0.001f64.log(1f64 / self.n as f64) as usize + 1;

        for _ in 0..k {
            if self.solve_once() {
                return true;
            }
        }

        false
    }

    /// Solves the problem.
    fn solve_once(&self) -> bool {
        let Some(count_combination) = CountCombination::new(self) else {
            return false;
        };

        // We rescale weights: set train_cost to 1 and car_cost to 0
        // this preserves the count combination. Budget is now equal to train_count.

        let wmax = 1;

        // check if field order is big enough
        assert!(FIELD_ORDER >= max(wmax * self.n + 1, self.n * self.n));

        let alpha = Matrix::new_uniform_random(self.n);

        fn h_matrix_det(
            y: Zp<FIELD_ORDER>,
            x: &Matrix<FIELD_ORDER>,
            edge_set: &HashMap<(usize, usize), usize>,
        ) -> Zp<FIELD_ORDER> {
            let mut h = Matrix::new(x.size());
            for i in 0..h.size() {
                for j in 0..h.size() {
                    h[(i, j)] = if let Some(cost) = edge_set.get(&(i, j)) {
                        y.pow(*cost) * x[(i, j)]
                    } else {
                        Zp::ZERO
                    };
                }
            }

            h.det()
        }

        let edge_set = self
            .connections
            .iter()
            .map(|c| {
                (
                    (c.volunteer, c.city),
                    if c.cost == self.train_cost { 1 } else { 0 },
                )
            })
            .collect::<HashMap<_, _>>();

        assert!(self.n * wmax + 1 < FIELD_ORDER);
        let gammas = (1..=self.n * wmax + 1).map(Zp::new).collect::<Vec<_>>();

        let p = {
            let mut p = Matrix::new(self.n * wmax + 1);
            for i in 0..p.size() {
                for j in 0..p.size() {
                    p[(i, j)] = gammas[i].pow(j);
                }
            }
            p
        };
        let r = gammas
            .into_iter()
            .map(|gamma| h_matrix_det(gamma, &alpha, &edge_set))
            .collect::<Vec<_>>();

        let c = LinearEquationSystem::new(p, r).solve();

        c[count_combination.train_count] != Zp::ZERO
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let iterator = stdin.lock().lines().collect::<Result<Vec<_>, _>>()?;

    let input = iterator.into_iter().collect::<Input>();

    if input.solve() {
        println!("yes");
    } else {
        println!("no")
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE_1: &str = "3 6 7 3 1
0 0 1
1 1 3
2 2 1
1 0 3
0 2 3
2 1 1
";
    const EXAMPLE_2: &str = "3 6 7 3 1
0 0 3
1 1 3
2 2 3
0 1 3
1 2 1
2 0 1
";
    const EXAMPLE_3: &str = "2 2 7 5 3
0 0 3
1 1 5
";

    fn get(s: &str) -> Input {
        s.lines().map(ToString::to_string).collect::<Input>()
    }

    #[test]
    fn parses_example_inputs() {
        let input = get(EXAMPLE_1);
        assert_eq!(
            input,
            Input {
                n: 3,
                budget: 7,
                train_cost: 3,
                car_cost: 1,
                connections: vec![
                    Connection {
                        volunteer: 0,
                        city: 0,
                        cost: 1
                    },
                    Connection {
                        volunteer: 1,
                        city: 1,
                        cost: 3
                    },
                    Connection {
                        volunteer: 2,
                        city: 2,
                        cost: 1
                    },
                    Connection {
                        volunteer: 1,
                        city: 0,
                        cost: 3
                    },
                    Connection {
                        volunteer: 0,
                        city: 2,
                        cost: 3
                    },
                    Connection {
                        volunteer: 2,
                        city: 1,
                        cost: 1
                    },
                ]
            }
        );

        let input = get(EXAMPLE_2);
        assert_eq!(
            input,
            Input {
                n: 3,
                budget: 7,
                train_cost: 3,
                car_cost: 1,
                connections: vec![
                    Connection {
                        volunteer: 0,
                        city: 0,
                        cost: 3
                    },
                    Connection {
                        volunteer: 1,
                        city: 1,
                        cost: 3
                    },
                    Connection {
                        volunteer: 2,
                        city: 2,
                        cost: 3
                    },
                    Connection {
                        volunteer: 0,
                        city: 1,
                        cost: 3
                    },
                    Connection {
                        volunteer: 1,
                        city: 2,
                        cost: 1
                    },
                    Connection {
                        volunteer: 2,
                        city: 0,
                        cost: 1
                    },
                ]
            }
        );

        let input = get(EXAMPLE_3);
        assert_eq!(
            input,
            Input {
                n: 2,
                budget: 7,
                train_cost: 5,
                car_cost: 3,
                connections: vec![
                    Connection {
                        volunteer: 0,
                        city: 0,
                        cost: 3
                    },
                    Connection {
                        volunteer: 1,
                        city: 1,
                        cost: 5
                    },
                ]
            },
        );
    }

    #[test]
    fn finds_correct_count_combinations() {
        let input = get(EXAMPLE_1);
        assert_eq!(
            CountCombination::new(&input),
            Some(CountCombination {
                train_count: 2,
                car_count: 1
            })
        );

        let input = get(EXAMPLE_2);
        assert_eq!(
            CountCombination::new(&input),
            Some(CountCombination {
                train_count: 2,
                car_count: 1
            })
        );

        let input = get(EXAMPLE_3);
        assert_eq!(CountCombination::new(&input), None);
    }

    #[test]
    fn field_order_is_prime() {
        for i in 2..FIELD_ORDER {
            assert!(FIELD_ORDER % i != 0);
        }
    }

    mod zp {
        use super::*;

        #[test]
        fn creation() {
            let a = Zp::<FIELD_ORDER>::new(FIELD_ORDER);
            let b = Zp::<FIELD_ORDER>::new(FIELD_ORDER + 1);

            assert_eq!(a, Zp::new(0));
            assert_eq!(b, Zp::new(1));
        }

        #[test]
        fn multiplicative_inverses() {
            let a = Zp::<FIELD_ORDER>::new(2);
            let b = Zp::<FIELD_ORDER>::new(3);

            assert_eq!(a.mutliplicative_inverse(), Zp::new(10006));
            assert_eq!(b.mutliplicative_inverse(), Zp::new(13341));
        }

        #[test]
        #[should_panic]
        fn zero_has_no_multiplicative_inverse() {
            Zp::<FIELD_ORDER>::ZERO.mutliplicative_inverse();
        }

        #[test]
        fn addition() {
            let a = Zp::<FIELD_ORDER>::new(1);
            let b = Zp::<FIELD_ORDER>::new(2);

            assert_eq!(a + b, Zp::new(3));
        }

        #[test]
        fn subtraction() {
            let a = Zp::<FIELD_ORDER>::new(1);
            let b = Zp::<FIELD_ORDER>::new(2);

            assert_eq!(a - b, Zp::new(FIELD_ORDER - 1));
        }

        #[test]
        fn multiplication() {
            let a = Zp::<FIELD_ORDER>::new(2);
            let b = Zp::<FIELD_ORDER>::new(FIELD_ORDER - 1);

            assert_eq!(a * b, Zp::new(FIELD_ORDER - 2));
        }

        #[test]
        fn division() {
            let a = Zp::<FIELD_ORDER>::new(6);
            let b = Zp::<FIELD_ORDER>::new(3);

            assert_eq!(a / b, Zp::new(2));
        }

        #[test]
        fn exponentiation() {
            let a = Zp::<FIELD_ORDER>::new(123);
            let b = Zp::<FIELD_ORDER>::new(3);

            assert_eq!(a.pow(321), Zp::new(12023));
            assert_eq!(b.pow(14), Zp::new(340));
            assert_eq!(a.pow(0), Zp::new(1));
            assert_eq!(a.pow(1), Zp::new(123));
        }
    }

    mod matrix {
        use super::*;

        #[test]
        fn creation() {
            let m = Matrix::<FIELD_ORDER>::new(3);

            assert_eq!(m.size(), 3);
            assert_eq!(m.data.len(), 9);
        }

        #[test]
        fn matrix_multiplication() {
            let a: Matrix<FIELD_ORDER> = Matrix::from([[1, 2], [3, 4]]);
            let b: Matrix<FIELD_ORDER> = Matrix::from([[5, 6], [7, 8]]);
            let c: Matrix<FIELD_ORDER> = Matrix::from([[19, 22], [43, 50]]);

            assert_eq!(a * b, c);
        }

        #[test]
        #[should_panic]
        fn matrix_multiplication_fails_for_different_size_matrices() {
            let a: Matrix<FIELD_ORDER> = Matrix::from([[1, 2], [3, 4]]);
            let b: Matrix<FIELD_ORDER> = Matrix::from([[5, 6, 8], [7, 8, 2], [0, 12, 9]]);

            let _ = a * b;
        }

        #[test]
        fn matrix_vector_multiplication() {
            let a: Matrix<FIELD_ORDER> = Matrix::from([[1, 2], [3, 4]]);
            let b: Vec<Zp<FIELD_ORDER>> = vec![Zp::new(5), Zp::new(7)];
            let c: Vec<Zp<FIELD_ORDER>> = vec![Zp::new(19), Zp::new(43)];

            assert_eq!(a * b, c);
        }

        #[test]
        #[should_panic]
        fn matrix_vector_multiplication_fails_for_different_sizes() {
            let a: Matrix<FIELD_ORDER> = Matrix::from([[1, 2], [3, 4]]);
            let b: Vec<Zp<FIELD_ORDER>> = vec![Zp::new(5), Zp::new(7), Zp::new(8)];

            let _ = a * b;
        }

        #[test]
        fn determinant() {
            let m: Matrix<FIELD_ORDER> = Matrix::from([[6, 18, 3], [2, 12, 1], [4, 15, 3]]);
            assert_eq!(m.det(), Zp::new(36));

            let m: Matrix<FIELD_ORDER> = Matrix::from([[0, 5108, 0], [0, 0, 4018], [6655, 0, 0]]);
            assert_eq!(m.det(), Zp::new(5852));
        }
    }

    mod linear_equation_system {
        use super::*;

        #[test]
        #[should_panic]
        fn linear_equation_system_fails_for_different_sizes() {
            let matrix: Matrix<FIELD_ORDER> = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
            let rhs = vec![Zp::new(1), Zp::new(2)];

            LinearEquationSystem::new(matrix, rhs);
        }
    }

    mod input {
        use super::*;

        #[test]
        fn solves_example_inputs() {
            let input = get(EXAMPLE_1);
            let result = input.solve();
            assert!(result);

            let input = get(EXAMPLE_2);
            let result = input.solve();
            assert!(!result);

            let input = get(EXAMPLE_3);
            let result = input.solve();
            assert!(!result);
        }

        fn shuffle<T>(vec: &mut [T]) {
            let mut rand = random_numbers();
            let n = vec.len();
            for i in 0..(n - 1) {
                let j = (rand() as usize) % (n - i) + i;
                vec.swap(i, j);
            }
        }

        #[test]
        fn solves_big_input() {
            let mut rand_int = random_numbers();

            let n = 100;
            let t = rand_int() as usize % 201;
            let c = rand_int() as usize % t;

            let mut cities = (0..n).collect::<Vec<_>>();
            shuffle(&mut cities);

            // construct perfect matching
            let mut budget = 0;
            let mut connections = vec![];
            for i in 0..n {
                let cost = if rand_int() % 2 == 0 { t } else { c };
                budget += cost;

                connections.push(Connection {
                    volunteer: i,
                    city: cities[i],
                    cost,
                });
            }

            let input = Input {
                n,
                budget,
                train_cost: t,
                car_cost: c,
                connections,
            };

            let result = input.solve();

            assert!(result);
        }
    }
}
