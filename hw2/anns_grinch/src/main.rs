#![allow(clippy::needless_range_loop)]

//! Solution by team:
//! - Marcin Wojnarowski (376886)
//! - Jonathan Arnoult (369910)
//! - Emilien Ganier (369941)

use std::io::{self, BufRead, Write};

struct Input {
    d: usize,
    r: usize,
    c: f64,
    n: usize,
    queries: usize,
    center: Vec<bool>,
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

        let center = iter
            .next()
            .unwrap()
            .split(' ')
            .map(|l| l == "1")
            .collect::<Vec<_>>();

        assert_eq!(center.len(), d);

        Self {
            d,
            r,
            c,
            n,
            queries,
            center,
        }
    }
}

impl Input {
    fn solve(&self) -> Option<Vec<bool>> {
        todo!()
    }
}

struct OnlineANNS {
    d: usize,
}

impl OnlineANNS {
    fn new(dim: usize) -> Self {
        Self { d: dim }
    }

    fn query(&self, q: &[bool]) -> Option<Vec<bool>> {
        assert_eq!(q.len(), self.d);
        println!(
            "q {}",
            q.iter()
                .map(|&b| if b { "1" } else { "b" })
                .collect::<Vec<_>>()
                .join(" ")
        );
        io::stdout().flush().unwrap();

        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        let mut response = line.trim().split(' ');
        let size: usize = response.next().unwrap().parse().unwrap();

        if size == 1 {
            return None;
        }

        assert!(size == self.d);

        let answer = response.map(|c| c == "1").collect::<Vec<_>>();

        assert!(answer.len() == self.d);

        Some(answer)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let iterator = stdin.lock().lines().collect::<Result<Vec<_>, _>>()?;

    let input = iterator.into_iter().collect::<Input>();

    if let Some(result) = input.solve() {
        println!(
            "* {}",
            result
                .iter()
                .map(|&b| if b { "1" } else { "0" })
                .collect::<Vec<_>>()
                .join(" ")
        );
    } else {
        panic!("Failed to solve");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE_INPUT: &str = "10 2 1.5 3 60
0 0 0 0 0 0 0 0 0 0
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
        assert_eq!(input.n, 3);
        assert_eq!(input.queries, 60);
        assert_eq!(input.center, vec![false; 10]);
    }
}
