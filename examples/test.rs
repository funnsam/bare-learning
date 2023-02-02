use bare_learning::{network::*, randomizer::init_randomizer, utils::*};

fn main() {
    init_randomizer();
    let mut nn = Network::new(1, &vec![2, 1]).unwrap();
    print_stuff(&nn, 0);
    for i in 1..=100 {
        let time = train(&mut nn).as_micros();
        print_stuff(&nn, i * 10000);
        println!("Training 10,000 times took {time}micro secs, avg: {}ms.", time / 10000)
    }
}

fn train(nn: &mut Network) -> std::time::Duration {
    let start = std::time::SystemTime::now();
    for _ in 0..10000 {
        nn.train(&vec![Data { inputs: vec![0.1, 0.3], outputs: vec![0.2] }, Data { inputs: vec![0.6, 0.9], outputs: vec![0.75] }, Data { inputs: vec![0.2, 0.4], outputs: vec![0.3] }], 0.15);
    }
    start.elapsed().unwrap()
}

fn print_stuff(nn: &Network, i: usize) {
    let pred = nn.predict(&vec![0.4, 0.6]).unwrap()[0];
    println!("\x1b[0;1m{i:010} ---- {:4.5}%\x1b[0m ---- {pred:03.25}", (1.0-loss(pred, 0.5)) * 100.0);
}
