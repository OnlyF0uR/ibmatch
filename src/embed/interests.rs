use once_cell::sync::Lazy;

pub const INTEREST_EMB_DIM: usize = 16;

const TOTAL_INTERESTS: usize = 142;

// Static interest embeddings (randomly initialized for now)
pub static INTEREST_EMBEDDINGS: Lazy<Vec<[f32; INTEREST_EMB_DIM]>> = Lazy::new(|| {
    (0..TOTAL_INTERESTS)
        .map(|_| {
            let mut v = [0.0; INTEREST_EMB_DIM];
            for i in 0..INTEREST_EMB_DIM {
                v[i] = rand::random::<f32>() - 0.5; // random in [-0.5,0.5]
            }
            // normalize vector
            let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
});

/// Convert interest IDs to dense embedding
pub fn interests_to_vector(ids: &[u32]) -> [f32; INTEREST_EMB_DIM] {
    let mut combined = [0f32; INTEREST_EMB_DIM];

    if ids.is_empty() {
        return combined;
    }

    for &id in ids {
        if let Some(emb) = INTEREST_EMBEDDINGS.get(id as usize) {
            for i in 0..INTEREST_EMB_DIM {
                combined[i] += emb[i];
            }
        }
    }

    let count = ids.len() as f32;
    for val in &mut combined {
        *val /= count;
    }

    combined
}
