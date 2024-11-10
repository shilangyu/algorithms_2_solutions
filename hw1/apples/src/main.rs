use std::{
    io::{self, BufRead},
    str::FromStr,
};

struct Input {
    alice: i32,
    bob: i32,
}

impl FromStr for Input {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (alice, bob) = s.split_once(' ').ok_or("incorrect format")?;
        let (alice, bob) = (alice.parse::<i32>()?, bob.parse::<i32>()?);

        Ok(Self { alice, bob })
    }
}

impl Input {
    fn solve(&self) -> i32 {
        self.alice + self.bob
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut iterator = stdin.lock().lines();
    let line = iterator.next().unwrap()?;

    let input = line.parse::<Input>()?;
    let result = input.solve();

    println!("{}", result);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    mod parsing {
        use super::*;

        #[test]
        fn correct_input() {
            let input = "1 2".parse::<Input>().unwrap();
            assert_eq!(input.alice, 1);
            assert_eq!(input.bob, 2);
        }

        #[test]
        fn incorrect_input() {
            let input = "1".parse::<Input>();
            assert!(input.is_err());
        }
    }

    mod solving {
        use super::*;

        #[test]
        fn simple() {
            let input = Input { alice: 1, bob: 2 };
            assert_eq!(input.solve(), 3);
        }
    }
}
