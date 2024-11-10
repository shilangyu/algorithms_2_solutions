use std::io::{self, BufRead};

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
        let mut header = header.split(" ");
        let (n, m, b, t, c) = (
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
        );

        let connections = iter
            .map(|line| {
                let mut line = line.split(" ");
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

impl Input {
    /// Solves the problem.
    fn solve(&self) -> bool {
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

    #[test]
    fn parses_example_inputs() {
        let input = EXAMPLE_1
            .lines()
            .map(ToString::to_string)
            .collect::<Input>();
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

        let input = EXAMPLE_2
            .lines()
            .map(ToString::to_string)
            .collect::<Input>();
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

        let input = EXAMPLE_3
            .lines()
            .map(ToString::to_string)
            .collect::<Input>();
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
}
