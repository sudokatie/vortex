// Collision detection module

pub mod aabb;
pub mod broadphase;
pub mod ccd;
pub mod contact;
pub mod epa;
pub mod gjk;
pub mod shapes;
pub mod simd;

pub use aabb::Aabb;
pub use broadphase::{BroadPhase, Bvh, SweepAndPrune, SpatialHash};
pub use ccd::{
    TimeOfImpact, SweepResult, calculate_toi, needs_ccd, sweep_test,
    sweep_sphere_box, sweep_sphere_capsule, sweep_box_box, sweep_capsule_capsule,
};
pub use contact::{ContactPoint, ContactManifold, generate_contacts, clip_polygon, reduce_manifold};
pub use epa::{PenetrationInfo, epa};
pub use gjk::{gjk_intersection, gjk_distance};
pub use shapes::{CollisionShape, Face};
pub use simd::{SimdVec3x4, SimdAabb4, simd_sphere_support, simd_box_support};
