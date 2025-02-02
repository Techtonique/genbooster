use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

pub fn create_rng(seed: u64) -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(seed)
} 