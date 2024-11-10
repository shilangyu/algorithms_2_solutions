use std::{
    cmp::max,
    io::{self, BufRead},
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Sub},
};

/// Rust std has no random generators. This is based on:
/// https://github.com/rust-lang/rust/blob/1.55.0/library/core/src/slice/sort.rs#L559-L573
fn random_numbers() -> impl Iterator<Item = u32> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let mut random = RandomState::new().build_hasher().finish() as u32;
    std::iter::repeat_with(move || {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        random
    })
}

/// A prime number used for Zp arithmetic. Number chosen to be big enough to handle the problem.
const FIELD_ORDER: usize = 20011;

/// Represents an element of the field Z/pZ for a prime number `P`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Zp<const P: usize>(usize);

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
}

/// Represents a square matrix over a Zp field.
struct Matrix<const P: usize> {
    data: Vec<Zp<P>>,
    // Invariant: size * size == data.len()
    size: usize,
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
        let data = vec![Zp::<P>::ZERO; size * size];

        Self { data, size }
    }

    fn new_uniform_random(size: usize) -> Self {
        let data = random_numbers()
            .take(size * size)
            .map(|n| Zp::<P>::new(n as _))
            .collect();

        Self { data, size }
    }

    const fn size(&self) -> usize {
        self.size
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

impl Input {
    /// Solves the problem.
    fn solve(self) -> bool {
        let Some(_) = CountCombination::new(&self) else {
            return false;
        };

        // TODO: rescale and shift to lower this number
        let wmax = max(self.train_cost, self.car_cost);

        // check if field order is big enough
        assert!(FIELD_ORDER >= max(wmax * self.n + 1, self.n * self.n));

        let alpha = Matrix::<FIELD_ORDER>::new_uniform_random(self.n);

        // TODO: is gamma allowed to be zero? At some point we are doing gamma^0, that would be undefined for gamma=0

        todo!()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let iterator = stdin.lock().lines().collect::<Result<Vec<_>, _>>()?;

    let input = iterator.into_iter().collect::<Input>();

    let result = input.solve();

    println!("{}", if result { "yes" } else { "no" });

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

            assert_eq!(a, Zp::<FIELD_ORDER>::new(0));
            assert_eq!(b, Zp::<FIELD_ORDER>::new(1));
        }

        #[test]
        fn multiplicative_inverses() {
            let a = Zp::<FIELD_ORDER>::new(2);
            let b = Zp::<FIELD_ORDER>::new(3);

            assert_eq!(a.mutliplicative_inverse(), Zp::<FIELD_ORDER>::new(10006));
            assert_eq!(b.mutliplicative_inverse(), Zp::<FIELD_ORDER>::new(13341));
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
            let c = Zp::<FIELD_ORDER>::new(3);

            assert_eq!(a + b, c);
        }

        #[test]
        fn subtraction() {
            let a = Zp::<FIELD_ORDER>::new(1);
            let b = Zp::<FIELD_ORDER>::new(2);
            let c = Zp::<FIELD_ORDER>::new(FIELD_ORDER - 1);

            assert_eq!(a - b, c);
        }

        #[test]
        fn multiplication() {
            let a = Zp::<FIELD_ORDER>::new(2);
            let b = Zp::<FIELD_ORDER>::new(FIELD_ORDER - 1);
            let c = Zp::<FIELD_ORDER>::new(FIELD_ORDER - 2);

            assert_eq!(a * b, c);
        }

        #[test]
        fn division() {
            let a = Zp::<FIELD_ORDER>::new(6);
            let b = Zp::<FIELD_ORDER>::new(3);
            let c = Zp::<FIELD_ORDER>::new(2);

            assert_eq!(a / b, c);
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
        fn lu_decompose() {
            // FIXME
        }
    }
}
