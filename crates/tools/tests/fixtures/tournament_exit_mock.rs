//! Windowsの終了観測・進捗統合試験だけに使うUSI fixture。
use std::io::{self, BufRead, Write};

fn main() {
    let mode = std::env::var("TOURNAMENT_EXIT_MODE").unwrap();
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        match line.as_str() {
            "usi" if mode == "startup_fail" => std::process::exit(7),
            "usi" => println!("id name exit-observation-fixture\nusiok"),
            "isready" => println!("readyok"),
            "quit" if mode == "nonzero" => std::process::exit(7),
            "quit" if mode == "ignore" => continue,
            "quit" => return,
            _ if line.starts_with("go ") => println!("bestmove resign"),
            _ => continue,
        }
        io::stdout().flush().unwrap();
    }
}
