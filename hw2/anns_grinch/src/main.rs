#![allow(clippy::needless_range_loop)]

//! Solution by team:
//! - Marcin Wojnarowski (376886)
//! - Jonathan Arnoult (369910)
//! - Emilien Ganier (369941)

use core::f64;
use std::io::{self, BufRead, Read, Write};

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

fn manhattan_distance(a: &[bool], b: &[bool]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

fn shuffle<T>(vec: &mut [T]) {
    let mut rand = random_numbers();
    let n = vec.len();
    for i in 0..(n - 1) {
        let j = (rand() as usize) % (n - i) + i;
        vec.swap(i, j);
    }
}

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
        // TODO: how many trails?
        let trials = 123;

        for _ in 0..trials {
            if let Some(result) = self.solve_once() {
                return Some(result);
            }
        }

        None
    }

    fn solve_once(&self) -> Option<Vec<bool>> {
        let anns = OnlineANNS::new(self.d);
        let mu = std::cmp::min(
            self.r,
            (2.0 * f64::consts::E * f64::consts::E * (f64::ln(self.n as f64) + 1.0)).ceil()
                as usize,
        );

        // sample q
        let mut indices = (0..self.d).collect::<Vec<_>>();
        shuffle(&mut indices);
        let mut q = self.center.clone();
        for i in indices.iter().take(self.r - mu) {
            q[*i] = !q[*i];
        }
        drop(indices);

        while anns.query(&q).is_some() && manhattan_distance(&q, &self.center) < self.r {
            let w =
                (self.c * self.r as f64).ceil() as usize + 1 - manhattan_distance(&q, &self.center);

            // sample I
            let mut indices = (0..self.d)
                .filter(|&i| q[i] == self.center[i])
                .collect::<Vec<_>>();
            shuffle(&mut indices);
            indices.truncate(w);

            // find j*
            let mut u = q.clone();
            let mut prev = None;
            for i in indices {
                u[i] = !u[i];
                if anns.query(&u).is_none() {
                    if let Some(j) = prev {
                        // TODO: what happened here with the type annotation?
                        q[j] = !(q[j] as bool);
                        break;
                    }
                }
                prev = Some(i);
            }
        }

        if anns.query(&q).is_none() && manhattan_distance(&q, &self.center) <= self.r {
            Some(q)
        } else {
            None
        }
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
                .map(|&b| if b { "1" } else { "0" })
                .collect::<Vec<_>>()
                .join(" ")
        );
        io::stdout().flush().unwrap();

        let mut buf = vec![];
        #[allow(clippy::unused_io_amount)]
        std::io::stdin().read(&mut buf).unwrap();
        let line = String::from_utf8(buf).unwrap();
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
        io::stdout().flush().unwrap();
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
