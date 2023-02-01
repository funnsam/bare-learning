use bare_learning::{network::*, randomizer::init_randomizer};

fn main() {
    init_randomizer();
    let mut nn = Network::new(3, &vec![2, 2, 2]).unwrap();
    println!("{:?}", nn);
    println!("{:?}", nn.predict(&vec![0.0_f32, 1.0_f32]));
    nn.train(&vec![Data { inputs: vec![0.1_f32, 0.3_f32], outputs: vec![0.2_f32, 0.4_f32] }, Data { inputs: vec![0.3_f32, 0.5_f32], outputs: vec![0.6_f32, 0.8_f32] }], 0.5);
    println!("{:?}", nn);
    println!("{:?}", nn.predict(&vec![0.0_f32, 1.0_f32]));
}
