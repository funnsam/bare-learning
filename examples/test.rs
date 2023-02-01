use bare_learning::{network::*, randomizer::init_randomizer};

fn main() {
    init_randomizer();
    let mut nn = Network::new(3, &vec![2, 2, 1]).unwrap();
    println!("{:?}", nn);
    println!("{:?}", nn.predict(&vec![0.0_f32, 1.0_f32]));
}
