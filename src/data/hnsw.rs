use hnsw_rs::prelude::*;
use once_cell::sync::Lazy;
use std::sync::{Arc, RwLock};

// Configure HNSW options
fn hnsw_options() -> (usize, usize, usize, usize, DistCosine) {
    let dim = 128; // embedding dimension
    let max_elements = 1_000_000;
    let max_conn0 = 16; // max connections for layer 0
    let max_conn = 16; // max connections for other layers
    (max_conn, max_elements, dim, max_conn0, DistCosine)
}

// Global HNSW instance wrapped in Arc
static HNSW_INDEX: Lazy<Arc<RwLock<Hnsw<'static, f32, DistCosine>>>> = Lazy::new(|| {
    let (max_conn, max_elements, dim, max_conn0, dist) = hnsw_options();
    let hnsw = Hnsw::<f32, DistCosine>::new(max_conn, max_elements, dim, max_conn0, dist);
    Arc::new(RwLock::new(hnsw))
});

// Access helper
pub fn get_hnsw_index() -> Arc<RwLock<Hnsw<'static, f32, DistCosine>>> {
    HNSW_INDEX.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_and_search() {
        let (max_conn, max_elements, dim, max_conn0, dist) = hnsw_options();
        let hnsw = Hnsw::<f32, DistCosine>::new(max_conn, max_elements, dim, max_conn0, dist);

        // Create more distinct test vectors
        let vec1: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let vec2: Vec<f32> = (0..128).map(|i| (i + 50) as f32 * 0.01).collect();
        let vec3: Vec<f32> = (0..128).map(|i| (i + 100) as f32 * 0.01).collect();

        // Insert vectors
        hnsw.insert((&vec1, 1));
        hnsw.insert((&vec2, 2));
        hnsw.insert((&vec3, 3));

        // Search for similar vectors
        let search_results = hnsw.search(&vec1, 3, 30);

        // Should find at least the vector itself
        assert!(!search_results.is_empty());

        // Check that our target vector (ID 1) is in the results
        let found_target = search_results.iter().any(|result| result.d_id == 1);
        assert!(
            found_target,
            "Target vector with ID 1 should be found in results"
        );

        // The closest match should have the smallest distance (ideally 0 for exact match)
        let closest_result = &search_results[0];
        assert!(
            closest_result.distance <= search_results[1].distance,
            "Results should be ordered by distance"
        );
    }

    #[test]
    fn test_global_hnsw_instance() {
        let hnsw = get_hnsw_index();

        // Acquire write lock and insert a vector
        {
            let hnsw_write = hnsw.write().unwrap();
            let vec: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
            hnsw_write.insert((&vec, 1));
        }

        // Acquire read lock and search for the vector
        {
            let hnsw_read = hnsw.read().unwrap();
            let vec: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
            let results = hnsw_read.search(&vec, 1, 30);
            assert!(!results.is_empty());
            assert_eq!(results[0].d_id, 1);
        }
    }
}
