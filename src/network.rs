use crate::randomizer::*;

#[derive(Debug, Clone)]
struct Layer {
    len: usize,
    p_len: usize,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>
}

impl Layer {
    fn new(size: usize, p_size: usize) -> Self {
        let mut weights = Vec::with_capacity(size);
        for _ in 0..p_size {
            weights.push(batch_f32(size, 1.0, 0.5))
        }

        Self {
            len: size,
            p_len: p_size,
            weights,
            bias: batch_f32(size, 1.0, 0.5)
        }
    }

    fn predict(&self, prev: &Vec<f32>) -> Vec<f32> {
        let mut results = Vec::new();
        for i in 0..self.len {
            let bias = self.bias[i];
            let mut sum = 0.0_f32;
            for j in 0..self.p_len {
                sum += prev[j] * self.weights[j][i];
            }
            results.push(sigmoid(bias + sum))
        }
        results
    }
}

/// This is a neural network.
#[derive(Debug)]
pub struct Network {
    input : Vec<f32>,
    hidden: Vec<Layer>,
    output: Layer
}

impl Network {
    /// Used for creating a neural network. Layers
    /// defines how many layers that a neural network
    /// contains (counts input and output). Sizes
    /// defines how large each layer is (also counts in
    /// and output layers).
    pub fn new(layers: usize, sizes: &Vec<usize>) -> Option<Self> {
        if layers != sizes.len() || layers < 2 || sizes.len() < 2 {
            return None
        }
        let mut netw = Self {
            input : batch_f32(sizes[0], 1.0, 0.5),
            hidden: Vec::with_capacity(sizes.len()),
            output: Layer::new(sizes[sizes.len()-1], sizes[sizes.len()-2])
        };
        for i in 0..layers-2 {
            netw.hidden.push(Layer::new(sizes[i], sizes[i+1]))
        }
        Some(netw)
    }

    /// Predict an answer.
    pub fn predict(&self, data: &Vec<f32>) -> Option<Vec<f32>> {
        if data.len() != self.input.len() {
            return None
        }

        let mut prev_iter = Vec::with_capacity(self.input.len());
        for (i, el) in self.input.iter().enumerate() {
            prev_iter.push(el + data[i])
        }

        for el in self.hidden.iter() {
            prev_iter = el.predict(&prev_iter);
        }

        Some(self.output.predict(&prev_iter))
    }

    /// Trains a neural network.
    pub fn train(&mut self, dataset: &Vec<Data>) {
        todo!()
    }
}

/// Contains input and expected output values. Used for
/// training [Network]
#[derive(Debug, Clone)]
pub struct Data {
    pub inputs : Vec<f32>,
    pub outputs: Vec<f32>
}

fn sigmoid(a: f32) -> f32 {
    1.0 / (1.0 + std::f32::consts::E.powf(a))
}
