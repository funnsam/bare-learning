use bare_learning::{network::*, randomizer::init_randomizer, utils::*};
use std::thread::*;

fn main() {
    init_randomizer();

    let mut handles = Vec::new();
    for i in 1..=10 {
        handles.push(
            spawn(move || {
                per_thread(i);
            })
        )
    }

    for _ in 0..handles.len() {
        handles.pop().unwrap().join().unwrap()
    }
}

fn per_thread(thread_id: usize) -> Network {
    let mut nn = Network::new(1, &vec![2, 1]).unwrap();
    print_stuff(&nn, 0, thread_id);
    for i in 1..=100 {
        let _ = train(&mut nn).as_micros();
        print_stuff(&nn, i * 10000, thread_id);
    }
    nn
}

fn train(nn: &mut Network) -> std::time::Duration {
    let start = std::time::SystemTime::now();
    for _ in 0..10000 {
        nn.train(&vec![Data { inputs: vec![0.1, 0.3], outputs: vec![0.2] }, Data { inputs: vec![0.6, 0.9], outputs: vec![0.75] }, Data { inputs: vec![0.2, 0.4], outputs: vec![0.3] }], 0.01);
    }
    start.elapsed().unwrap()
}

fn print_stuff(nn: &Network, i: usize, n: usize) {
    let pred = nn.predict(&vec![0.4, 0.6]).unwrap()[0];
    println!("\x1b[0;1;32m{n:02}: \x1b[0;1m{i:010} ---- {:4.5}%\x1b[0m ---- {pred:03.5}", (1.0-loss(pred, 0.5)) * 100.0);
}
