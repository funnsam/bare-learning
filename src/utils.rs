pub fn loss(a: f32, b: f32) -> f32 {
    (a - b).abs()
}

pub fn error(a: &Vec<f32>, b: &Vec<f32>) -> Option<f32> {
    if a.len() != b.len() {
        return None
    }

    let mut t = 0.0;
    for (i, el) in a.iter().enumerate() {
        t += loss(*el, b[i])
    }
    Some(t / a.len() as f32)
}
