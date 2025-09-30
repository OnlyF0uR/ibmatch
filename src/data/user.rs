use std::sync::Arc;

use bincode::{
    Decode, Encode,
    config::{self},
};
use hnsw_rs::hnsw::FilterT;
use rocksdb::DB;

use crate::{
    data::hnsw,
    embed::{self, INTEREST_EMB_DIM},
};
use crate::{embed::TEXT_EMB_DIM, errors::MatchError};

/// Filter implementation for HNSW search
struct UserFilter {
    allowed_ids: std::collections::HashSet<usize>,
}

impl FilterT for UserFilter {
    fn hnsw_filter(&self, id: &hnsw_rs::prelude::DataId) -> bool {
        self.allowed_ids.contains(&id)
    }
}

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
    pub user_id: u32,
    pub age: u8,
    pub gender: u8, // Gender index (0=M, 1=F, 2=O)

    pub likeness_score: f32,   // Overall likeness score (0.0 to 1.0)
    pub preference_score: f32, // Overall preference score (0.0 to 1.0)
    pub norm_rating: f32,      // normalized rating score (0.0 to 1.0)

    pub likeness_updates: u32,   // Number of updates to likeness score
    pub preference_updates: u32, // Number of updates to preference score

    pub location: [f64; 2],       // lat/lon
    pub preferences: Preferences, // Matching preference / filters
    pub meta: Meta,
    pub multiplier: f32,                     // exposure / scoring multiplier
    pub text_embedding: [f32; TEXT_EMB_DIM], // 50-d embedding for bio/interests
    pub interest_embeddings: [f32; INTEREST_EMB_DIM],
}

impl UserProfile {
    /// Function to create a new user (we shall infer embeddings)
    pub fn create_new(
        db: &Arc<DB>,
        user_id: u32,
        age: u8,
        gender: u8,
        location: [f64; 2],
        preferences: Preferences,
        raw_intersts: &[u32],
        raw_biography: &str,
    ) -> Result<Self, MatchError> {
        // 1. Calculate text embedding
        let (t_embed, i_embed) =
            embed::calculate_persistent_embeddings(raw_intersts, raw_biography)?;

        // Get current unix timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // 2. Create user profile with default values
        let user = UserProfile {
            user_id,
            age,
            gender,
            likeness_score: 0.5,   // Neutral initial likeness score
            preference_score: 0.5, // Neutral initial preference score
            norm_rating: 0.5,
            likeness_updates: 0,
            preference_updates: 0,
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
        };

        // 3. Store the user in the DB and HNSW
        let key = format!("user:{}", user.user_id);
        let value = user.encode()?;

        db.put(key.as_bytes(), &value)?;

        let likeness_embedding = embed::likeness_to_vector(user.likeness_score);
        let combined = embed::combine_embeddings(
            &user.text_embedding,
            &user.interest_embeddings,
            &likeness_embedding,
        )?;

        // Add the user into the HNSW index
        let hnsw = hnsw::get_hnsw_index();
        let hnsw_write = hnsw.write()?;
        hnsw_write.insert((&combined, user.user_id as usize));

        // Add user to gender mapping
        hnsw::add_user_to_gender(user.user_id as usize, user.gender);

        Ok(user)
    }

    /// Load a user
    pub fn load_user(db: &Arc<DB>, user_id: u32) -> Result<Self, MatchError> {
        let key = format!("user:{}", user_id);

        let value = db.get(key.as_bytes())?;
        let parsed = match value {
            Some(v) => UserProfile::decode(&v)?,
            None => return Err(MatchError::UserNotFound),
        };

        // Combine the relevant embeddings
        let likeness_embedding = embed::likeness_to_vector(parsed.likeness_score);
        let combined = embed::combine_embeddings(
            &parsed.text_embedding,
            &parsed.interest_embeddings,
            &likeness_embedding,
        )?;

        // Add to HNSW to maintain accurate representation (will override if already exists)
        let hnsw = hnsw::get_hnsw_index();
        let hnsw_write = hnsw.write()?;
        hnsw_write.insert((&combined, user_id as usize));

        // Add to gender mapping
        hnsw::add_user_to_gender(user_id as usize, parsed.gender);

        Ok(parsed)
    }

    /// Search
    pub fn search(&self, db: &Arc<DB>, top_k: usize) -> Result<Vec<UserProfile>, MatchError> {
        // It is vital that we substitute the likeness embedding with the preference embedding
        // because we search towards likeness in accordance with the preferences of this user
        let preferece_embedding = embed::likeness_to_vector(self.preference_score);

        let _combined = embed::combine_embeddings(
            &self.text_embedding,
            &self.interest_embeddings,
            &preferece_embedding,
        )?;

        // Get the allowed gender ids
        // these are the ids of users of which we can prefilter on gender
        let allowed_ids =
            hnsw::get_allowed_ids(db, self.user_id as usize, &self.preferences.gender);

        let hnsw = hnsw::get_hnsw_index();
        let hnsw_read = hnsw.read()?;

        let filter = UserFilter { allowed_ids };
        let search = hnsw_read.search_filter(&self.text_embedding, top_k * 5, 16, Some(&filter));

        // Here we have the IDs of the candidates
        let mut users = Vec::new();
        for n in search {
            let candidate = UserProfile::load_user(&db, n.d_id as u32)?;
            if self.post_filter(&candidate) {
                users.push(candidate);
            }
        }

        Ok(users)
    }

    // We can update the gender preference directly by updating it in
    // rocksdb.
    pub fn update_gender_preference(&mut self, _new_genders: Vec<u8>) -> Result<(), MatchError> {
        todo!()
    }

    /// Update likeness score
    /// This calculates a new likeness score. The total number of updates is taken into account,
    /// so that changes are less impactful over time. Supports both likes and dislikes as
    /// indicated by the `positive` parameter.
    pub fn update_likeness_score(&mut self, _positive: bool) -> Result<(), MatchError> {
        todo!()
    }

    /// Update preference score
    /// This calculates a new preference score. This is to be executed upon a swipe by
    /// the current user. Based on the likeness score of the liked/disliked user, we adjust
    /// our own preference score. The total number of updates is taken into account,
    /// so that changes are less impactful over time. Supports both likes and dislikes as
    /// indicated by the `positive` parameter.
    pub fn update_preference_score(
        &mut self,
        _others_likeness: f32,
        _positive: bool,
    ) -> Result<(), MatchError> {
        todo!()
    }

    fn post_filter(&self, _candidate: &UserProfile) -> bool {
        // TODO: implement filtering based on age range, distance, rating, plan, banned status etc.

        // Also handle recent swipes
        true
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

                        let likeness_embedding =
                            embed::likeness_to_vector(user_profile.likeness_score);
                        let combined = embed::combine_embeddings(
                            &user_profile.text_embedding,
                            &user_profile.interest_embeddings,
                            &likeness_embedding,
                        )?;

                        // Insert text embedding into HNSW index
                        let hnsw_write = hnsw.write()?;
                        hnsw_write.insert((&combined, user_id));

                        // Add user to gender mapping
                        hnsw::add_user_to_gender(user_id, user_profile.gender);

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
            user_id: 1,
            age: 24,
            gender: 1,
            likeness_score: 0.5,
            preference_score: 0.5,
            norm_rating: 0.75,
            likeness_updates: 1,
            preference_updates: 5,
            location: [52.3702, 4.895],
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
        assert_eq!(decoded.user_id, user.user_id);
        assert_eq!(decoded.age, user.age);
        assert_eq!(decoded.gender, user.gender);
        assert_eq!(decoded.likeness_score, user.likeness_score);
        assert_eq!(decoded.preference_score, user.preference_score);
        assert_eq!(decoded.norm_rating, user.norm_rating);
        assert_eq!(decoded.likeness_updates, user.likeness_updates);
        assert_eq!(decoded.preference_updates, user.preference_updates);
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
            user_id: 1,
            age: 30,
            gender: 2,
            likeness_score: 0.5,
            preference_score: 0.5,
            norm_rating: 0.85,
            likeness_updates: 2,
            preference_updates: 3,
            location: [40.7128, -74.0060],
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

        assert_eq!(decoded.age, user.age);
        assert_eq!(decoded.preferences.gender.len(), 3);
        assert_eq!(decoded.preferences.gender, user.preferences.gender);
        assert_eq!(decoded.meta.plan, user.meta.plan);
        assert_eq!(decoded.meta.banned, user.meta.banned);
        assert_eq!(decoded.text_embedding, user.text_embedding);
        assert_eq!(decoded.interest_embeddings, user.interest_embeddings);
    }

    #[test]
    fn test_empty_gender_preferences() {
        let user = UserProfile {
            user_id: 1,
            age: 25,
            gender: 0,
            likeness_score: 0.5,
            preference_score: 0.5,
            norm_rating: 0.5,
            likeness_updates: 0,
            preference_updates: 0,
            location: [0.0, 0.0],
            preferences: Preferences {
                gender: vec![],
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
            user_id: u32::MAX,
            age: 255,  // Max u8
            gender: 0, // Empty string
            likeness_score: 1.0,
            preference_score: 0.0,
            norm_rating: 0.0, // Min normalized rating
            likeness_updates: u32::MAX,
            preference_updates: u32::MAX,
            location: [-90.0, -180.0], // Min lat/lon
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

        assert_eq!(decoded.user_id, user.user_id);
        assert_eq!(decoded.age, 255);
        assert_eq!(decoded.gender, 0);
        assert_eq!(decoded.likeness_score, 1.0);
        assert_eq!(decoded.preference_score, 0.0);
        assert_eq!(decoded.norm_rating, 0.0);
        assert_eq!(decoded.likeness_updates, u32::MAX);
        assert_eq!(decoded.preference_updates, u32::MAX);
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
