use std::io::{self, Read};

fn main() {
    let mut stdin = io::stdin().lock();
    let mut buf = Vec::with_capacity(256);
    let _ = stdin.read_to_end(&mut buf);

    let mut is_alice = true;
    let mut alice = 0;
    let mut bob = 0;

    for d in buf.into_iter() {
        if d > b'9' || d < b'0' {
            is_alice = false;
            continue;
        }

        if is_alice {
            alice = alice * 10 + (d - b'0') as i32;
        } else {
            bob = bob * 10 + (d - b'0') as i32;
        }
    }

    println!("{}", alice + bob);
}
