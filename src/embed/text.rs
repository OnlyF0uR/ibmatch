use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use dashmap::DashMap;
use once_cell::sync::Lazy;

use crate::errors::MatchError;

pub const TEXT_EMB_DIM: usize = 50;

/// Global embeddings map using DashMap for concurrent access
static GLOVE_EMBEDDINGS: Lazy<DashMap<String, [f32; TEXT_EMB_DIM]>> = Lazy::new(|| DashMap::new());

/// Initialize embeddings from a given path
pub fn initialize_embeddings(path: impl Into<PathBuf>) -> Result<(), MatchError> {
    let path = path.into();
    let file = File::open(&path)?;
    let reader = BufReader::new(file);

    let mut map = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        let mut iter = line.split_whitespace();
        if let Some(word) = iter.next() {
            let vec_res: Result<Vec<f32>, _> = iter.map(|x| x.parse::<f32>()).collect();
            if let Ok(vec) = vec_res {
                if vec.len() == TEXT_EMB_DIM {
                    let mut arr = [0f32; TEXT_EMB_DIM];
                    arr.copy_from_slice(&vec);
                    map.insert(word.to_string(), arr);
                }
            }
        }
    }

    // Clear and populate DashMap
    GLOVE_EMBEDDINGS.clear();
    for (key, value) in map {
        GLOVE_EMBEDDINGS.insert(key, value);
    }

    Ok(())
}

/// Fast concurrent lookup for text embeddings
fn get_embedding(word: &str) -> Option<[f32; TEXT_EMB_DIM]> {
    GLOVE_EMBEDDINGS.get(word).map(|entry| *entry.value())
}

pub fn text_to_embedding(text: &str) -> [f32; TEXT_EMB_DIM] {
    let mut vec_sum = [0f32; TEXT_EMB_DIM];
    let mut count = 0;

    for token in text
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|s| !s.is_empty())
    {
        if let Some(embed) = get_embedding(token) {
            for i in 0..TEXT_EMB_DIM {
                vec_sum[i] += embed[i];
            }
            count += 1;
        }
    }

    if count > 0 {
        for i in 0..TEXT_EMB_DIM {
            vec_sum[i] /= count as f32;
        }
    }

    // Normalize
    let norm = vec_sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for i in 0..TEXT_EMB_DIM {
            vec_sum[i] /= norm;
        }
    }

    vec_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_embedding() {
        // Initialize embeddings with test data path
        let embeddings_path =
            "embeddings/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt";
        let ir = initialize_embeddings(embeddings_path);
        assert!(ir.is_ok());

        let bio = "I love hiking, movies, and coding!";
        let embedding = text_to_embedding(bio);

        assert_eq!(embedding.len(), 50);
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0);
    }
}
