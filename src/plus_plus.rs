use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    Rng, RngExt,
};

/// k-means++ centroid initialization.
///
/// # Panics
///
/// Panics if buffer is empty.
///
/// # Reference
///
/// Based on Section 2.2 from `k-means++: The Advantages of Careful Seeding` by
/// Arthur and Vassilvitskii (2007).
pub fn init_plus_plus<C: crate::Calculate + Clone>(
    k: usize,
    rng: &mut impl Rng,
    buf: &[C],
    centroids: &mut Vec<C>,
) {
    if k == 0 {
        return;
    }
    let len = buf.len();
    assert!(len > 0);

    // Choose first centroid at random, uniform sampling from input buffer
    centroids.push(buf[rng.random_range(0..len)].clone());
    if k == 1 {
        return;
    }

    // Keep raw squared distances so only the newest center needs comparing.
    let mut distances = vec![f32::MAX; len];
    // Honor any centers supplied by callers before initialization.
    for cent in &centroids[..centroids.len() - 1] {
        let _ = crate::kernels::update_distances(buf, cent, &mut distances);
    }

    // Pick a new centroid with weighted probability of `D(x)^2 / sum(D(x)^2)`,
    // where `D(x)^2` is the distance to the closest centroid
    for _ in 1..k {
        let newest = centroids.last().unwrap();
        let sum = crate::kernels::update_distances(buf, newest, &mut distances);

        // If centroids match all colors, return early
        if !sum.is_normal() {
            return;
        }

        // Preserve normalized weights for reproducible seeded selection, while
        // retaining raw distances and avoiding a separate weights buffer.
        let sampler = WeightedIndex::new(distances.iter().map(|distance| *distance / sum)).unwrap();
        centroids.push(buf[sampler.sample(rng)].clone());
    }
}
