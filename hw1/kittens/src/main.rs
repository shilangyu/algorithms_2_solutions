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
    b: usize,
    t: usize,
    c: usize,
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
            b,
            t,
            c,
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

    mod parsing {
        use super::*;

        #[test]
        fn test_input() {
            let input = Input {
                n: 3,
                b: 1,
                t: 1,
                c: 2,
                connections: vec![
                    Connection {
                        volunteer: 0,
                        city: 1,
                        cost: 1,
                    },
                    Connection {
                        volunteer: 1,
                        city: 2,
                        cost: 2,
                    },
                ],
            };

            let input_str = "3 2 1 1 2\n0 1 1\n1 2 2\n"
                .lines()
                .map(|s| s.to_string())
                .collect::<Input>();

            assert_eq!(input, input_str);
        }
    }
}
