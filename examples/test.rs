use bare_learning::{network::*, randomizer::init_randomizer};

fn main() {
    init_randomizer();
    let mut nn = Network::new(1, &vec![2, 1]).unwrap();
    for i in 0..10000 {
        nn.train(&vec![Data { inputs: vec![0.1_f32, 0.3_f32], outputs: vec![0.2_f32] }, Data { inputs: vec![0.6_f32, 0.9_f32], outputs: vec![0.75_f32] }], 1.0);
        println!("\n---------------------{i}\n{:?}", nn);
        print!("{:?}", nn.predict(&vec![0.1_f32, 0.3_f32]).unwrap());
        println!("{:?}", nn.predict(&vec![0.6_f32, 0.9_f32]).unwrap());
    }
}
