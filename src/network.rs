use crate::randomizer::*;
use crate::utils::*;

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
            bias: vec![0.0; size]
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


    // (a, z)
    fn pred_w_z(&self, prev: &Vec<f32>) -> Vec<(f32, f32)> {
        let mut results = Vec::new();
        for i in 0..self.len {
            let bias = self.bias[i];
            let mut sum = 0.0_f32;
            for j in 0..self.p_len {
                sum += prev[j] * self.weights[j][i];
            }
            results.push((sigmoid(bias + sum), bias + sum))
        }
        results
    }

    fn train(&mut self, data: &Vec<Data>, alpha: f32) {
        let mut results = Vec::with_capacity(data.len());
        for i in data.iter() {
            results.push(self.pred_w_z(&i.inputs));
        }

        let _expect = data.iter().map(|a| a.outputs.to_owned()).collect::<Vec<Vec<f32>>>();
        let _expect = _expect.iter().cloned().flat_map(|a| a.iter().cloned().collect::<Vec<_>>()).collect::<Vec<f32>>();
        let mut expect: Vec<Vec<f32>> = vec![Vec::with_capacity(_expect.len() >> 1); 2];

        for (i, el) in _expect.iter().enumerate() {
            expect[i%self.len].push(*el)
        }

        for node in self.weights.iter_mut() {
            for (j, weight) in node.iter_mut().enumerate() {
                let mut t = 0.0;
                for (i, r) in results.iter().enumerate() {
                    t += loss(r[j].0, expect[j][i]) * r[j].0 * r[j].1
                }
                *weight -= alpha * (t / *weight)
            }
        }

        for (j, bias) in self.bias.iter_mut().enumerate() {
            let mut t = 0.0;
            for r in results.iter() {
                t += r[j].1
            }
            *bias -= alpha * t;
        }
    }
}

/// This is a neural network.
#[derive(Debug)]
pub struct Network {
    hidden: Vec<Layer>,
    output: Layer
}

impl Network {
    /// Used for creating a neural network. Layers
    /// defines how many layers that a neural network
    /// contains (counts input). Sizes
    /// defines how large each layer is (also counts input layers).
    pub fn new(layers: usize, sizes: &Vec<usize>) -> Option<Self> {
        if layers != sizes.len()-1 || layers < 1 || sizes.len() < 2 {
            return None
        }
        let mut netw = Self {
            hidden: Vec::with_capacity(sizes.len()),
            output: Layer::new(sizes[sizes.len()-1], sizes[sizes.len()-2])
        };
        for i in 0..layers-1 {
            netw.hidden.push(Layer::new(sizes[i], sizes[i+1]))
        }
        Some(netw)
    }

    /// Predict an answer.
    pub fn predict(&self, data: &Vec<f32>) -> Option<Vec<f32>> {
        // if data.len() != self.input.len() {
        //     return None
        // }

        let mut prev_iter = data.clone();
        for el in self.hidden.iter() {
            prev_iter = el.predict(&prev_iter);
        }

        Some(self.output.predict(&prev_iter))
    }

    /// Trains a neural network. Alpha is the learning rate
    pub fn train(&mut self, data: &Vec<Data>, alpha: f32) -> Option<()> {
        if data[0].outputs.len() != self.output.len {
            return None
        }

        self.output.train(data, alpha);

        Some(())
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
    1.0 / (1.0 + std::f32::consts::E.powf(-a))
}
