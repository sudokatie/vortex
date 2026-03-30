//! Collision detection tests
//!
//! These tests require the `dim3` feature (enabled by default).

#![cfg(feature = "dim3")]

use glam::Vec3;
use vortex::collision::*;
use vortex::math::Transform;

// =============================================================================
// AABB Tests
// =============================================================================

#[test]
fn test_aabb_new() {
    let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
    assert_eq!(aabb.min, Vec3::ZERO);
    assert_eq!(aabb.max, Vec3::ONE);
}

#[test]
fn test_aabb_from_center_extents() {
    let aabb = Aabb::from_center_extents(Vec3::ONE, Vec3::ONE);
    assert_eq!(aabb.min, Vec3::ZERO);
    assert_eq!(aabb.max, Vec3::splat(2.0));
}

#[test]
fn test_aabb_intersects() {
    let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
    let b = Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5));
    let c = Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0));
    
    assert!(a.intersects(&b));
    assert!(!a.intersects(&c));
}

#[test]
fn test_aabb_contains_point() {
    let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
    
    assert!(aabb.contains_point(Vec3::splat(0.5)));
    assert!(!aabb.contains_point(Vec3::splat(2.0)));
}

#[test]
fn test_aabb_merged() {
    let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
    let b = Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0));
    let merged = a.merged(&b);
    
    assert_eq!(merged.min, Vec3::ZERO);
    assert_eq!(merged.max, Vec3::splat(3.0));
}

#[test]
fn test_aabb_center() {
    let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(2.0));
    assert_eq!(aabb.center(), Vec3::ONE);
}

#[test]
fn test_aabb_extents() {
    let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(2.0));
    assert_eq!(aabb.extents(), Vec3::ONE);
}

// =============================================================================
// Shape Tests
// =============================================================================

#[test]
fn test_sphere_support() {
    let sphere = CollisionShape::sphere(2.0);
    
    let support_x = sphere.support(Vec3::X);
    assert!((support_x - Vec3::new(2.0, 0.0, 0.0)).length() < 0.001);
    
    let support_neg = sphere.support(Vec3::NEG_Y);
    assert!((support_neg - Vec3::new(0.0, -2.0, 0.0)).length() < 0.001);
}

#[test]
fn test_box_support() {
    let box_shape = CollisionShape::cube(Vec3::new(1.0, 2.0, 3.0));
    
    let support = box_shape.support(Vec3::ONE);
    assert_eq!(support, Vec3::new(1.0, 2.0, 3.0));
    
    let support_neg = box_shape.support(-Vec3::ONE);
    assert_eq!(support_neg, Vec3::new(-1.0, -2.0, -3.0));
}

#[test]
fn test_capsule_support() {
    let capsule = CollisionShape::capsule(1.0, 2.0);
    
    let support_up = capsule.support(Vec3::Y);
    assert!((support_up.y - 3.0).abs() < 0.001); // half_height + radius
    
    let support_down = capsule.support(Vec3::NEG_Y);
    assert!((support_down.y - (-3.0)).abs() < 0.001);
}

#[test]
fn test_convex_support() {
    let vertices = vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(0.0, -1.0, 1.0),
    ];
    let convex = CollisionShape::convex_hull(vertices);
    
    let support_up = convex.support(Vec3::Y);
    assert!((support_up.y - 1.0).abs() < 0.001);
}

#[test]
fn test_shape_aabb() {
    let sphere = CollisionShape::sphere(1.0);
    let aabb = sphere.local_aabb();
    assert_eq!(aabb.min, Vec3::splat(-1.0));
    assert_eq!(aabb.max, Vec3::splat(1.0));
}

#[test]
fn test_shape_inertia_sphere() {
    let sphere = CollisionShape::sphere(1.0);
    let inertia = sphere.inertia_tensor(1.0);
    // I = 2/5 * m * r^2 = 0.4 for unit sphere with unit mass
    assert!((inertia.x_axis.x - 0.4).abs() < 0.001);
}

// =============================================================================
// GJK Tests
// =============================================================================

#[test]
fn test_gjk_spheres_intersecting() {
    let sphere = CollisionShape::sphere(1.0);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(1.5, 0.0, 0.0));
    
    assert!(gjk_intersection(&sphere, &t1, &sphere, &t2).is_some());
}

#[test]
fn test_gjk_spheres_not_intersecting() {
    let sphere = CollisionShape::sphere(1.0);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));
    
    assert!(gjk_intersection(&sphere, &t1, &sphere, &t2).is_none());
}

#[test]
fn test_gjk_boxes_intersecting() {
    let box_shape = CollisionShape::cube(Vec3::ONE);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(1.5, 0.0, 0.0));
    
    assert!(gjk_intersection(&box_shape, &t1, &box_shape, &t2).is_some());
}

#[test]
fn test_gjk_boxes_not_intersecting() {
    let box_shape = CollisionShape::cube(Vec3::ONE);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));
    
    assert!(gjk_intersection(&box_shape, &t1, &box_shape, &t2).is_none());
}

#[test]
fn test_gjk_sphere_box() {
    let sphere = CollisionShape::sphere(1.0);
    let box_shape = CollisionShape::cube(Vec3::ONE);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(1.5, 0.0, 0.0));
    
    assert!(gjk_intersection(&sphere, &t1, &box_shape, &t2).is_some());
}

#[test]
fn test_gjk_distance() {
    let sphere = CollisionShape::sphere(1.0);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(5.0, 0.0, 0.0));
    
    let dist = gjk_distance(&sphere, &t1, &sphere, &t2);
    // Distance should be 5 - 2 = 3 (minus two radii)
    assert!((dist - 3.0).abs() < 0.1);
}

#[test]
fn test_gjk_touching_spheres() {
    let sphere = CollisionShape::sphere(1.0);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(2.0, 0.0, 0.0));
    
    let dist = gjk_distance(&sphere, &t1, &sphere, &t2);
    assert!(dist < 0.1); // Should be ~0
}

// =============================================================================
// EPA Tests
// =============================================================================

#[test]
fn test_epa_spheres() {
    let sphere = CollisionShape::sphere(1.0);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));
    
    if let Some(simplex) = gjk_intersection(&sphere, &t1, &sphere, &t2) {
        if let Some(info) = epa(simplex, &sphere, &t1, &sphere, &t2) {
            // Penetration should be about 1.0
            assert!(info.depth > 0.5 && info.depth < 1.5);
            // Normal should point along X axis
            assert!(info.normal.x.abs() > 0.9);
        }
    }
}

#[test]
fn test_epa_boxes() {
    let box_shape = CollisionShape::cube(Vec3::ONE);
    let t1 = Transform::from_position(Vec3::ZERO);
    let t2 = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));
    
    if let Some(simplex) = gjk_intersection(&box_shape, &t1, &box_shape, &t2) {
        if let Some(info) = epa(simplex, &box_shape, &t1, &box_shape, &t2) {
            assert!(info.depth > 0.5);
        }
    }
}

// =============================================================================
// Broadphase Tests
// =============================================================================

#[test]
fn test_sap_basic() {
    use slotmap::SlotMap;
    use vortex::world::BodyHandle;
    
    let mut sap = SweepAndPrune::new();
    let mut map: SlotMap<BodyHandle, ()> = SlotMap::with_key();
    
    let h1 = map.insert(());
    let h2 = map.insert(());
    
    sap.insert(h1, Aabb::new(Vec3::ZERO, Vec3::ONE));
    sap.insert(h2, Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5)));
    
    let pairs = sap.query_pairs();
    assert_eq!(pairs.len(), 1);
}

#[test]
fn test_bvh_basic() {
    use slotmap::SlotMap;
    use vortex::world::BodyHandle;
    
    let mut bvh = Bvh::new();
    let mut map: SlotMap<BodyHandle, ()> = SlotMap::with_key();
    
    let h1 = map.insert(());
    let h2 = map.insert(());
    
    bvh.insert(h1, Aabb::new(Vec3::ZERO, Vec3::ONE));
    bvh.insert(h2, Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5)));
    
    let pairs = bvh.query_pairs();
    assert_eq!(pairs.len(), 1);
}

#[test]
fn test_spatial_hash_basic() {
    use slotmap::SlotMap;
    use vortex::world::BodyHandle;
    
    let mut hash = SpatialHash::new(1.0);
    let mut map: SlotMap<BodyHandle, ()> = SlotMap::with_key();
    
    let h1 = map.insert(());
    let h2 = map.insert(());
    
    hash.insert(h1, Aabb::new(Vec3::ZERO, Vec3::ONE));
    hash.insert(h2, Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5)));
    
    let pairs = hash.query_pairs();
    assert_eq!(pairs.len(), 1);
}

// =============================================================================
// Contact Generation Tests
// =============================================================================

#[test]
fn test_contact_manifold_new() {
    use vortex::world::BodyHandle;
    use slotmap::SlotMap;
    
    let mut map: SlotMap<BodyHandle, ()> = SlotMap::with_key();
    let h1 = map.insert(());
    let h2 = map.insert(());
    
    let manifold = ContactManifold::new(h1, h2);
    assert_eq!(manifold.body_a, h1);
    assert_eq!(manifold.body_b, h2);
    assert!(manifold.contacts.is_empty());
}

#[test]
fn test_contact_point_creation() {
    let point = ContactPoint::new(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.9, 0.0, 0.0),
        Vec3::X,
        0.1,
    );
    
    assert_eq!(point.local_a, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(point.local_b, Vec3::new(0.9, 0.0, 0.0));
    assert_eq!(point.penetration, 0.1);
}

#[test]
fn test_manifold_add_point() {
    use vortex::world::BodyHandle;
    use slotmap::SlotMap;
    
    let mut map: SlotMap<BodyHandle, ()> = SlotMap::with_key();
    let h1 = map.insert(());
    let h2 = map.insert(());
    
    let mut manifold = ContactManifold::new(h1, h2);
    manifold.add_point(ContactPoint::new(
        Vec3::ZERO, Vec3::ZERO, Vec3::Y, 0.1
    ));
    
    assert_eq!(manifold.contacts.len(), 1);
}

// =============================================================================
// SIMD Batch Tests
// =============================================================================

#[test]
fn test_simd_vec3x4_dot() {
    use vortex::collision::SimdVec3x4;
    
    let a = SimdVec3x4::new(
        Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE,
    );
    let b = SimdVec3x4::new(
        Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE,
    );
    
    let dots = a.dot(&b);
    
    assert!((dots[0] - 1.0).abs() < 0.001);
    assert!((dots[1] - 1.0).abs() < 0.001);
    assert!((dots[2] - 1.0).abs() < 0.001);
    assert!((dots[3] - 3.0).abs() < 0.001);
}

#[test]
fn test_simd_aabb4_intersects() {
    use vortex::collision::SimdAabb4;
    
    let aabbs = SimdAabb4 {
        min_x: [0.0, 0.0, 2.0, 10.0],
        min_y: [0.0, 0.0, 2.0, 10.0],
        min_z: [0.0, 0.0, 2.0, 10.0],
        max_x: [1.0, 1.0, 3.0, 11.0],
        max_y: [1.0, 1.0, 3.0, 11.0],
        max_z: [1.0, 1.0, 3.0, 11.0],
    };
    
    // Test AABB overlapping [0] and [1]
    let mask = aabbs.intersects_single(
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(1.5, 1.5, 1.5),
    );
    
    assert!(mask & 0b0001 != 0); // overlaps [0]
    assert!(mask & 0b0010 != 0); // overlaps [1]
    assert!(mask & 0b0100 == 0); // no overlap [2]
    assert!(mask & 0b1000 == 0); // no overlap [3]
}
