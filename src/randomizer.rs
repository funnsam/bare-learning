pub static mut SEED: u32 = 0;

pub fn batch_f32(n: usize, l: f32, m: f32) -> Vec<f32> {
    let mut a = Vec::with_capacity(n);
    for _ in 0..n {
        a.push((rand_f32() % l) - m)
    }
    a
}

pub fn rand_f32() -> f32 {
    unsafe {
        let mut x = SEED;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        SEED = x;
        x as i32 as f32 / 100000000.0
    }
}

#[cfg(feature = "std")]
pub fn init_randomizer() {
    unsafe {
        use std::time::*;
        SEED = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as u32;
    }
}
