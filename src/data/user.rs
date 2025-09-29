use std::sync::Arc;

use bincode::{
    Decode, Encode,
    config::{self},
};
use rocksdb::DB;

use crate::{
    combine_embeddings,
    data::hnsw,
    embed::{self, INTEREST_EMB_DIM},
};
use crate::{embed::TEXT_EMB_DIM, errors::MatchError};

/// Represents a user's preferences for matching
#[derive(Debug, Encode, Decode)]
pub struct Preferences {
    pub gender: Vec<u8>, // Gender indeces the user is interested in (0=M, 1=F, 2=O)
    pub age_range: [u8; 2], // min and max age
    pub distance_km: u32, // maximum distance
}

/// Metadata for internal tracking
#[derive(Debug, Encode, Decode)]
pub struct Meta {
    pub last_seen: u64,         // Unix timestamp
    pub impressions_today: u32, // exposure count today
    pub plan: u8,               // 0=free, 1=premium
    pub banned: bool,
}

/// Full user profile with multiplier for boosting
#[derive(Debug, Encode, Decode)]
pub struct UserProfile {
    pub age: u8,
    pub gender: u8,               // Gender index (0=M, 1=F, 2=O)
    pub location: [f64; 2],       // lat/lon
    pub preferences: Preferences, // Matching preference / filters
    pub meta: Meta,
    pub multiplier: f32,                     // exposure / scoring multiplier
    pub text_embedding: [f32; TEXT_EMB_DIM], // 50-d embedding for bio/interests
    pub interest_embeddings: [f32; INTEREST_EMB_DIM],
    pub norm_rating: f32, // normalized rating score (0.0 to 1.0)
}

impl UserProfile {
    /// Function to create a new user (we shall infer embeddings)
    pub fn create_new(
        age: u8,
        gender: u8,
        location: [f64; 2],
        preferences: Preferences,
        raw_intersts: &[u32],
        raw_biography: &str,
    ) -> Result<Self, MatchError> {
        // 1. Calculate text embedding
        let (t_embed, i_embed) = embed::calculate_embeddings(raw_intersts, raw_biography)?;

        // Get current unix timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // 2. Create user profile with default values
        let user = UserProfile {
            age,
            gender,
            location,
            preferences,
            meta: Meta {
                last_seen: now,
                impressions_today: 0,
                plan: 0,
                banned: false,
            },
            multiplier: 1.0,
            text_embedding: t_embed,
            interest_embeddings: i_embed,
            norm_rating: 0.5,
        };

        Ok(user)
    }

    /// Store a user
    pub fn store_user(db: &Arc<DB>, user_id: u32, user: &UserProfile) -> Result<(), MatchError> {
        let key = format!("user:{}", user_id);
        let value = user.encode()?;

        db.put(key.as_bytes(), &value)?;

        // Add the user into the HNSW index
        let hnsw = hnsw::get_hnsw_index();
        let hnsw_write = hnsw.write()?;
        hnsw_write.insert((&user.text_embedding, user_id as usize));

        Ok(())
    }

    /// Load a user
    pub fn load_user(db: &Arc<DB>, user_id: u32) -> Result<Self, MatchError> {
        let key = format!("user:{}", user_id);

        let value = db.get(key.as_bytes())?;
        let parsed = match value {
            Some(v) => UserProfile::decode(&v)?,
            None => return Err(MatchError::UserNotFound),
        };

        // Add to HNSW to maintain accurate representation (will override if already exists)
        let hnsw = hnsw::get_hnsw_index();
        let hnsw_write = hnsw.write()?;
        hnsw_write.insert((&parsed.text_embedding, user_id as usize));

        Ok(parsed)
    }

    /// Search
    pub fn search(&self) -> Result<Vec<UserProfile>, MatchError> {
        // Now note that self is the user that is searching
        // so we can filter based on preferences
        // We can combine embeddings using embed::combine_embeddings
        // And use those as a basis for the search

        let _combined =
            combine_embeddings(&self.text_embedding, &self.interest_embeddings, (0.4, 0.5))?;

        // TODO: Implement this
        todo!();
    }

    /// Encode the UserProfile to a byte vector using bincode
    fn encode(&self) -> Result<Vec<u8>, MatchError> {
        let config = config::standard();
        let encoded: Vec<u8> = bincode::encode_to_vec(self, config)?;

        Ok(encoded)
    }

    /// Decode a UserProfile from a byte slice
    fn decode(bytes: &[u8]) -> Result<Self, MatchError> {
        let config = config::standard();
        let (decoded, _): (Self, _) = bincode::decode_from_slice(bytes, config)?;

        Ok(decoded)
    }

    pub fn normalize_rating(mut rating: f32) -> f32 {
        if rating < 1.0 {
            rating = 1.0;
        } else if rating > 10.0 {
            rating = 10.0;
        }

        (rating as f32 - 1.0) / 9.0
    }
}

pub fn bulk_load(
    db: &Arc<DB>,
    hnsw: &Arc<
        std::sync::RwLock<hnsw_rs::prelude::Hnsw<'static, f32, hnsw_rs::prelude::DistCosine>>,
    >,
) -> Result<usize, MatchError> {
    let iter = db.iterator(rocksdb::IteratorMode::Start);
    let mut loaded_count = 0;

    for item in iter {
        let (key, value) = item?;

        // Check if this is a user key (format: "user:{id}")
        if let Ok(key_str) = std::str::from_utf8(&key) {
            if key_str.starts_with("user:") {
                // Extract user ID from key
                if let Some(id_str) = key_str.strip_prefix("user:") {
                    if let Ok(user_id) = id_str.parse::<usize>() {
                        // Decode user profile
                        let user_profile = UserProfile::decode(&value)?;

                        // Insert text embedding into HNSW index
                        let hnsw_write = hnsw.write()?;
                        hnsw_write.insert((&user_profile.text_embedding, user_id));
                        loaded_count += 1;
                    }
                }
            }
        }
    }

    Ok(loaded_count)
}

// Test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let user = UserProfile {
            age: 24,
            gender: 1,
            location: [52.3702, 4.895],
            norm_rating: 0.75,
            preferences: Preferences {
                gender: vec![0],
                age_range: [22, 28],
                distance_km: 50,
            },
            meta: Meta {
                last_seen: 1_695_900_000,
                impressions_today: 3,
                plan: 1,
                banned: false,
            },
            multiplier: 1.2,
            text_embedding: [0.1; TEXT_EMB_DIM],
            interest_embeddings: [0.2; INTEREST_EMB_DIM],
        };

        // Test encoding
        let encoded = user.encode().expect("Failed to encode user");
        assert!(!encoded.is_empty());

        // Test decoding
        let decoded = UserProfile::decode(&encoded).expect("Failed to decode user");

        // Verify all fields match
        assert_eq!(decoded.age, user.age);
        assert_eq!(decoded.gender, user.gender);
        assert_eq!(decoded.location, user.location);
        assert_eq!(decoded.preferences.gender, user.preferences.gender);
        assert_eq!(decoded.preferences.age_range, user.preferences.age_range);
        assert_eq!(
            decoded.preferences.distance_km,
            user.preferences.distance_km
        );
        assert_eq!(decoded.meta.last_seen, user.meta.last_seen);
        assert_eq!(decoded.meta.impressions_today, user.meta.impressions_today);
        assert_eq!(decoded.meta.plan, user.meta.plan);
        assert_eq!(decoded.meta.banned, user.meta.banned);
        assert_eq!(decoded.multiplier, user.multiplier);
        assert_eq!(decoded.text_embedding, user.text_embedding);
        assert_eq!(decoded.interest_embeddings, user.interest_embeddings);
    }

    #[test]
    fn test_multiple_gender_preferences() {
        let user = UserProfile {
            age: 30,
            gender: 2,
            location: [40.7128, -74.0060], // NYC coordinates
            norm_rating: 0.85,
            preferences: Preferences {
                gender: vec![0, 1, 2],
                age_range: [18, 65],
                distance_km: 100,
            },
            meta: Meta {
                last_seen: 1_700_000_000,
                impressions_today: 0,
                plan: 0,
                banned: false,
            },
            multiplier: 0.8,
            text_embedding: [0.4; TEXT_EMB_DIM],
            interest_embeddings: [0.5; INTEREST_EMB_DIM],
        };

        let encoded = user.encode().expect("Failed to encode user");
        let decoded = UserProfile::decode(&encoded).expect("Failed to decode user");

        assert_eq!(decoded.preferences.gender.len(), 3);
        assert_eq!(decoded.preferences.gender, user.preferences.gender);
        assert_eq!(decoded.meta.plan, user.meta.plan);
        assert_eq!(decoded.meta.banned, user.meta.banned);
    }

    #[test]
    fn test_empty_gender_preferences() {
        let user = UserProfile {
            age: 25,
            gender: 0,
            location: [0.0, 0.0],
            norm_rating: 0.5,
            preferences: Preferences {
                gender: vec![], // Empty preferences
                age_range: [18, 99],
                distance_km: 1,
            },
            meta: Meta {
                last_seen: 0,
                impressions_today: 0,
                plan: 0,
                banned: true,
            },
            multiplier: 0.1,
            text_embedding: [0.0; TEXT_EMB_DIM],
            interest_embeddings: [0.0; INTEREST_EMB_DIM],
        };

        let encoded = user.encode().expect("Failed to encode user");
        let decoded = UserProfile::decode(&encoded).expect("Failed to decode user");

        assert_eq!(decoded.preferences.gender.len(), 0);
        assert!(decoded.meta.banned);
    }

    #[test]
    fn test_boundary_values() {
        let user = UserProfile {
            age: 255,                  // Max u8
            gender: 0,                 // Empty string
            location: [-90.0, -180.0], // Min lat/lon
            norm_rating: 0.0,          // Min normalized rating
            preferences: Preferences {
                gender: vec![0],
                age_range: [0, 255],   // Full u8 range
                distance_km: u32::MAX, // Max distance
            },
            meta: Meta {
                last_seen: u64::MAX,
                impressions_today: u32::MAX,
                plan: 1,
                banned: false,
            },
            multiplier: f32::MAX,
            text_embedding: [f32::MAX; TEXT_EMB_DIM],
            interest_embeddings: [f32::MIN; INTEREST_EMB_DIM],
        };

        let encoded = user.encode().expect("Failed to encode user");
        let decoded = UserProfile::decode(&encoded).expect("Failed to decode user");

        assert_eq!(decoded.age, 255);
        assert_eq!(decoded.gender, 0);
        assert_eq!(decoded.preferences.distance_km, u32::MAX);
        assert_eq!(decoded.meta.last_seen, u64::MAX);
        assert_eq!(decoded.multiplier, f32::MAX);
        assert_eq!(decoded.text_embedding, user.text_embedding);
        assert_eq!(decoded.interest_embeddings, user.interest_embeddings);
    }

    #[test]
    fn test_decode_invalid_data() {
        let invalid_data = vec![0xFF, 0xFF, 0xFF]; // Invalid bincode data
        let result = UserProfile::decode(&invalid_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_empty_data() {
        let empty_data = vec![];
        let result = UserProfile::decode(&empty_data);
        assert!(result.is_err());
    }
}
