#![allow(clippy::needless_range_loop)]

//! Solution by team:
//! - Marcin Wojnarowski (376886)
//! - Jonathan Arnoult (369910)
//! - Emilien Ganier (369941)

use std::io::{self, BufRead};

struct Input {
    d: usize,
    r: usize,
    c: f64,
    queries: usize,
    kids: Vec<Vec<bool>>,
}

impl FromIterator<String> for Input {
    /// Given a list of stdin lines, parse them into an `Input` struct.
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        let mut iter = iter.into_iter();

        let header = iter.next().unwrap();
        let mut header = header.split(' ');
        let (d, r, c, n, queries) = (
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
            header.next().unwrap().parse().unwrap(),
        );

        let kids = iter
            .map(|line| {
                let line = line.split(' ').map(|l| l == "1").collect::<Vec<_>>();

                assert!(line.len() == d);
                line
            })
            .collect::<Vec<_>>();

        assert_eq!(kids.len(), n);

        Self {
            d,
            r,
            c,
            queries,
            kids,
        }
    }
}

impl Input {
    fn solve(self) -> bool {
        todo!()
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

    const EXAMPLE_INPUT: &str = "10 2 1.5 3 60
0 0 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 1
1 1 1 0 1 0 1 1 1 1
";

    fn get(s: &str) -> Input {
        s.lines().map(ToString::to_string).collect::<Input>()
    }

    #[test]
    fn parses_example_input() {
        let input = get(EXAMPLE_INPUT);

        assert_eq!(input.d, 10);
        assert_eq!(input.r, 2);
        assert_eq!(input.c, 1.5);
        assert_eq!(input.queries, 60);
        assert_eq!(input.kids.len(), 3);
    }
}
