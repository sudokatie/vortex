//! Property-based tests for collision detection and physics.
//!
//! Uses proptest to verify invariants that should hold for all inputs.

#![allow(dead_code)]

use proptest::prelude::*;
use glam::Vec3;
use vortex::collision::{CollisionShape, gjk_intersection, gjk_distance, Aabb};
use vortex::math::Transform;

// =============================================================================
// Strategies for generating test data
// =============================================================================

fn _vec3_strategy() -> impl Strategy<Value = Vec3> {
    (-100.0f32..100.0, -100.0f32..100.0, -100.0f32..100.0)
        .prop_map(|(x, y, z)| Vec3::new(x, y, z))
}

fn small_vec3_strategy() -> impl Strategy<Value = Vec3> {
    (-10.0f32..10.0, -10.0f32..10.0, -10.0f32..10.0)
        .prop_map(|(x, y, z)| Vec3::new(x, y, z))
}

fn positive_f32() -> impl Strategy<Value = f32> {
    0.1f32..10.0
}

fn sphere_strategy() -> impl Strategy<Value = CollisionShape> {
    positive_f32().prop_map(CollisionShape::sphere)
}

fn box_strategy() -> impl Strategy<Value = CollisionShape> {
    (positive_f32(), positive_f32(), positive_f32())
        .prop_map(|(x, y, z)| CollisionShape::cube(Vec3::new(x, y, z)))
}

fn capsule_strategy() -> impl Strategy<Value = CollisionShape> {
    (positive_f32(), positive_f32())
        .prop_map(|(r, h)| CollisionShape::capsule(r, h))
}

fn shape_strategy() -> impl Strategy<Value = CollisionShape> {
    prop_oneof![
        sphere_strategy(),
        box_strategy(),
        capsule_strategy(),
    ]
}

fn _transform_strategy() -> impl Strategy<Value = Transform> {
    small_vec3_strategy().prop_map(Transform::from_position)
}

// =============================================================================
// GJK Properties
// =============================================================================

proptest! {
    /// GJK is symmetric: intersect(A, B) == intersect(B, A)
    #[test]
    fn gjk_symmetric(
        shape_a in shape_strategy(),
        shape_b in shape_strategy(),
        pos_a in small_vec3_strategy(),
        pos_b in small_vec3_strategy(),
    ) {
        let t_a = Transform::from_position(pos_a);
        let t_b = Transform::from_position(pos_b);
        
        let result_ab = gjk_intersection(&shape_a, &t_a, &shape_b, &t_b);
        let result_ba = gjk_intersection(&shape_b, &t_b, &shape_a, &t_a);
        
        prop_assert_eq!(result_ab.is_some(), result_ba.is_some());
    }
    
    /// Distance is symmetric: dist(A, B) == dist(B, A)
    #[test]
    fn gjk_distance_symmetric(
        shape_a in shape_strategy(),
        shape_b in shape_strategy(),
        pos_a in small_vec3_strategy(),
        pos_b in small_vec3_strategy(),
    ) {
        let t_a = Transform::from_position(pos_a);
        let t_b = Transform::from_position(pos_b);
        
        let dist_ab = gjk_distance(&shape_a, &t_a, &shape_b, &t_b);
        let dist_ba = gjk_distance(&shape_b, &t_b, &shape_a, &t_a);
        
        prop_assert!((dist_ab - dist_ba).abs() < 0.01);
    }
    
    /// Distance is non-negative
    #[test]
    fn gjk_distance_non_negative(
        shape_a in shape_strategy(),
        shape_b in shape_strategy(),
        pos_a in small_vec3_strategy(),
        pos_b in small_vec3_strategy(),
    ) {
        let t_a = Transform::from_position(pos_a);
        let t_b = Transform::from_position(pos_b);
        
        let dist = gjk_distance(&shape_a, &t_a, &shape_b, &t_b);
        
        prop_assert!(dist >= 0.0);
    }
    
    /// Same shape at same position intersects
    #[test]
    fn gjk_same_position_intersects(
        shape in shape_strategy(),
        pos in small_vec3_strategy(),
    ) {
        let t = Transform::from_position(pos);
        
        let result = gjk_intersection(&shape, &t, &shape, &t);
        
        prop_assert!(result.is_some());
    }
    
    /// Far apart shapes don't intersect
    #[test]
    fn gjk_far_apart_no_intersection(
        radius_a in 0.1f32..2.0,
        radius_b in 0.1f32..2.0,
    ) {
        let shape_a = CollisionShape::sphere(radius_a);
        let shape_b = CollisionShape::sphere(radius_b);
        
        // Place spheres far apart (more than sum of radii)
        let t_a = Transform::from_position(Vec3::ZERO);
        let t_b = Transform::from_position(Vec3::new(radius_a + radius_b + 10.0, 0.0, 0.0));
        
        let result = gjk_intersection(&shape_a, &t_a, &shape_b, &t_b);
        
        prop_assert!(result.is_none());
    }
    
    /// Overlapping spheres always intersect
    #[test]
    fn gjk_overlapping_spheres_intersect(
        radius_a in 0.5f32..2.0,
        radius_b in 0.5f32..2.0,
        overlap in 0.1f32..0.5,
    ) {
        let shape_a = CollisionShape::sphere(radius_a);
        let shape_b = CollisionShape::sphere(radius_b);
        
        // Place spheres so they overlap by 'overlap' amount
        let separation = radius_a + radius_b - overlap;
        let t_a = Transform::from_position(Vec3::ZERO);
        let t_b = Transform::from_position(Vec3::new(separation, 0.0, 0.0));
        
        let result = gjk_intersection(&shape_a, &t_a, &shape_b, &t_b);
        
        prop_assert!(result.is_some());
    }
}

// =============================================================================
// AABB Properties
// =============================================================================

proptest! {
    /// AABB intersection is symmetric
    #[test]
    fn aabb_intersection_symmetric(
        min_a in small_vec3_strategy(),
        size_a in (0.1f32..5.0, 0.1f32..5.0, 0.1f32..5.0),
        min_b in small_vec3_strategy(),
        size_b in (0.1f32..5.0, 0.1f32..5.0, 0.1f32..5.0),
    ) {
        let max_a = min_a + Vec3::new(size_a.0, size_a.1, size_a.2);
        let max_b = min_b + Vec3::new(size_b.0, size_b.1, size_b.2);
        
        let aabb_a = Aabb::new(min_a, max_a);
        let aabb_b = Aabb::new(min_b, max_b);
        
        prop_assert_eq!(aabb_a.intersects(&aabb_b), aabb_b.intersects(&aabb_a));
    }
    
    /// AABB always contains its center
    #[test]
    fn aabb_contains_center(
        min in small_vec3_strategy(),
        size in (0.1f32..5.0, 0.1f32..5.0, 0.1f32..5.0),
    ) {
        let max = min + Vec3::new(size.0, size.1, size.2);
        let aabb = Aabb::new(min, max);
        let center = aabb.center();
        
        prop_assert!(aabb.contains_point(center));
    }
    
    /// AABB merged with itself equals itself
    #[test]
    fn aabb_merge_self(
        min in small_vec3_strategy(),
        size in (0.1f32..5.0, 0.1f32..5.0, 0.1f32..5.0),
    ) {
        let max = min + Vec3::new(size.0, size.1, size.2);
        let aabb = Aabb::new(min, max);
        let merged = aabb.merged(&aabb);
        
        prop_assert!((merged.min - aabb.min).length() < 0.001);
        prop_assert!((merged.max - aabb.max).length() < 0.001);
    }
    
    /// Merged AABB contains both originals
    #[test]
    fn aabb_merge_contains_both(
        min_a in small_vec3_strategy(),
        size_a in (0.1f32..5.0, 0.1f32..5.0, 0.1f32..5.0),
        min_b in small_vec3_strategy(),
        size_b in (0.1f32..5.0, 0.1f32..5.0, 0.1f32..5.0),
    ) {
        let max_a = min_a + Vec3::new(size_a.0, size_a.1, size_a.2);
        let max_b = min_b + Vec3::new(size_b.0, size_b.1, size_b.2);
        
        let aabb_a = Aabb::new(min_a, max_a);
        let aabb_b = Aabb::new(min_b, max_b);
        let merged = aabb_a.merged(&aabb_b);
        
        prop_assert!(merged.contains_aabb(&aabb_a));
        prop_assert!(merged.contains_aabb(&aabb_b));
    }
}

// =============================================================================
// Shape Properties
// =============================================================================

proptest! {
    /// Support function returns point on or inside shape
    #[test]
    fn support_point_valid(
        radius in 0.1f32..5.0,
        dir_x in -1.0f32..1.0,
        dir_y in -1.0f32..1.0,
        dir_z in -1.0f32..1.0,
    ) {
        let shape = CollisionShape::sphere(radius);
        let dir = Vec3::new(dir_x, dir_y, dir_z);
        
        if dir.length() > 0.01 {
            let support = shape.support(dir.normalize());
            // Support point should be on sphere surface
            prop_assert!((support.length() - radius).abs() < 0.01);
        }
    }
    
    /// Box support is always at a corner
    #[test]
    fn box_support_at_corner(
        hx in 0.1f32..5.0,
        hy in 0.1f32..5.0,
        hz in 0.1f32..5.0,
        dir_x in -1.0f32..1.0,
        dir_y in -1.0f32..1.0,
        dir_z in -1.0f32..1.0,
    ) {
        let shape = CollisionShape::cube(Vec3::new(hx, hy, hz));
        let dir = Vec3::new(dir_x, dir_y, dir_z);
        
        if dir.length() > 0.01 {
            let support = shape.support(dir.normalize());
            // Support should be at one of the 8 corners
            prop_assert!((support.x.abs() - hx).abs() < 0.001);
            prop_assert!((support.y.abs() - hy).abs() < 0.001);
            prop_assert!((support.z.abs() - hz).abs() < 0.001);
        }
    }
}

// =============================================================================
// Physics Invariants
// =============================================================================

proptest! {
    /// Integration preserves momentum (no external forces)
    #[test]
    fn integration_conserves_momentum(
        mass in 0.1f32..10.0,
        vx in -10.0f32..10.0,
        vy in -10.0f32..10.0,
        vz in -10.0f32..10.0,
        dt in 0.001f32..0.1,
    ) {
        use vortex::dynamics::{IntegratorType, IntegrationState, IntegrationInput, integrate_state};
        
        let velocity = Vec3::new(vx, vy, vz);
        let state = IntegrationState::new(Vec3::ZERO, velocity);
        let input = IntegrationInput::linear(Vec3::ZERO); // No acceleration
        
        let new_state = integrate_state(IntegratorType::SemiImplicitEuler, state, input, dt);
        
        // Velocity should be unchanged (momentum = m*v conserved)
        prop_assert!((new_state.velocity - velocity).length() < 0.001);
    }
    
    /// Position changes by velocity * dt (no acceleration)
    #[test]
    fn integration_position_change(
        px in -10.0f32..10.0,
        py in -10.0f32..10.0,
        pz in -10.0f32..10.0,
        vx in -10.0f32..10.0,
        vy in -10.0f32..10.0,
        vz in -10.0f32..10.0,
        dt in 0.001f32..0.1,
    ) {
        use vortex::dynamics::{IntegratorType, IntegrationState, IntegrationInput, integrate_state};
        
        let position = Vec3::new(px, py, pz);
        let velocity = Vec3::new(vx, vy, vz);
        let state = IntegrationState::new(position, velocity);
        let input = IntegrationInput::linear(Vec3::ZERO);
        
        let new_state = integrate_state(IntegratorType::SemiImplicitEuler, state, input, dt);
        
        let expected_pos = position + velocity * dt;
        prop_assert!((new_state.position - expected_pos).length() < 0.01);
    }
}
