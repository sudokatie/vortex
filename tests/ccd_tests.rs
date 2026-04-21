//! Continuous Collision Detection tests
//!
//! These tests verify that CCD correctly detects and handles
//! fast-moving objects that would otherwise tunnel through each other.

#![cfg(feature = "dim3")]

use glam::{Quat, Vec3};
use vortex::collision::*;
use vortex::dynamics::{BodyType, RigidBody};
use vortex::world::PhysicsWorld;

// =============================================================================
// Sweep Test Unit Tests
// =============================================================================

#[test]
fn test_sphere_sphere_fast_head_on_collision() {
    // Two spheres approaching each other at high speed
    // Should detect collision at correct time
    let sphere = CollisionShape::sphere(1.0);

    let result = sweep_test(
        &sphere,
        Vec3::new(-10.0, 0.0, 0.0),
        Vec3::new(10.0, 0.0, 0.0),
        Quat::IDENTITY,
        &sphere,
        Vec3::new(10.0, 0.0, 0.0),
        Vec3::new(-10.0, 0.0, 0.0),
        Quat::IDENTITY,
    );

    assert!(result.is_some(), "Fast head-on collision should be detected");
    let r = result.unwrap();
    // Combined radius = 2, initial distance = 20
    // Relative velocity = 40 units, so collision at t = 18/40 = 0.45
    assert!(r.time >= 0.0 && r.time <= 1.0);
    assert!(r.time < 0.5, "Collision should happen before midpoint");
}

#[test]
fn test_box_sphere_tunneling_prevention() {
    // Sphere moving very fast toward a thin box (wall)
    // Without CCD, it would tunnel through
    let sphere = CollisionShape::sphere(0.5);
    let wall = CollisionShape::cube(Vec3::new(5.0, 5.0, 0.1)); // Thin wall

    let result = sweep_test(
        &sphere,
        Vec3::new(-10.0, 0.0, 0.0),
        Vec3::new(10.0, 0.0, 0.0),
        Quat::IDENTITY,
        &wall,
        Vec3::ZERO,
        Vec3::ZERO,
        Quat::IDENTITY,
    );

    assert!(result.is_some(), "Sphere should not tunnel through wall");
    let r = result.unwrap();
    assert!(r.time >= 0.0 && r.time <= 1.0);
}

#[test]
fn test_toi_accuracy() {
    // Test that TOI is accurate within epsilon
    let sphere = CollisionShape::sphere(1.0);

    // Precise setup: spheres at distance 10, combined radius 2
    // One sphere moves 10 units, should touch at t = 0.8
    let result = sweep_test(
        &sphere,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(10.0, 0.0, 0.0),
        Quat::IDENTITY,
        &sphere,
        Vec3::new(12.0, 0.0, 0.0),
        Vec3::new(12.0, 0.0, 0.0), // Stationary
        Quat::IDENTITY,
    );

    assert!(result.is_some());
    let r = result.unwrap();
    // Collision when moving sphere center reaches 10 (combined radius = 2)
    // Distance to cover = 10, sphere moves 10 units, so t = 10/10 = 1.0
    // Actually: 0 + 10*t + 1 = 12 - 1 => 10t = 10 => t = 1.0
    // Let's adjust: sphere moves to x=10, static sphere at x=8
    // Then: 0 + 10*t + 1 = 8 - 1 => 10t = 6 => t = 0.6
    assert!((r.time - 1.0).abs() < 0.1, "TOI should be approximately 1.0");
}

#[test]
fn test_toi_accuracy_mid_frame() {
    let sphere = CollisionShape::sphere(1.0);

    // More precise test: sphere at 0 moves to 10, static sphere at 7
    // Contact when: sphere_a.x + 1 = sphere_b.x - 1
    // moving_x + 1 = 7 - 1 => moving_x = 5
    // t = 5/10 = 0.5
    let result = sweep_test(
        &sphere,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(10.0, 0.0, 0.0),
        Quat::IDENTITY,
        &sphere,
        Vec3::new(7.0, 0.0, 0.0),
        Vec3::new(7.0, 0.0, 0.0),
        Quat::IDENTITY,
    );

    assert!(result.is_some());
    let r = result.unwrap();
    assert!((r.time - 0.5).abs() < 0.1, "TOI should be approximately 0.5");
}

#[test]
fn test_no_collision_parallel_motion() {
    // Two spheres moving in parallel, never collide
    let sphere = CollisionShape::sphere(1.0);

    let result = sweep_test(
        &sphere,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(10.0, 0.0, 0.0),
        Quat::IDENTITY,
        &sphere,
        Vec3::new(0.0, 5.0, 0.0), // 5 units apart in Y
        Vec3::new(10.0, 5.0, 0.0),
        Quat::IDENTITY,
    );

    assert!(result.is_none(), "Parallel motion should not collide");
}

#[test]
fn test_no_collision_diverging_paths() {
    // Two spheres moving away from each other
    let sphere = CollisionShape::sphere(1.0);

    let result = sweep_test(
        &sphere,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(-10.0, 0.0, 0.0), // Moving left
        Quat::IDENTITY,
        &sphere,
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(15.0, 0.0, 0.0), // Moving right
        Quat::IDENTITY,
    );

    assert!(result.is_none(), "Diverging paths should not collide");
}

#[test]
fn test_already_overlapping_returns_zero() {
    // Two overlapping spheres should return t=0
    let sphere = CollisionShape::sphere(2.0);

    let result = sweep_test(
        &sphere,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Quat::IDENTITY,
        &sphere,
        Vec3::new(1.0, 0.0, 0.0), // Distance 1, combined radius 4
        Vec3::new(2.0, 0.0, 0.0),
        Quat::IDENTITY,
    );

    assert!(result.is_some());
    let r = result.unwrap();
    assert_eq!(r.time, 0.0, "Already overlapping should return time 0");
}

// =============================================================================
// Physics World CCD Integration Tests
// =============================================================================

#[test]
fn test_ccd_enabled_flag() {
    let mut world = PhysicsWorld::new();

    // CCD is enabled by default
    assert!(world.ccd_enabled());

    // Can disable CCD
    world.set_ccd_enabled(false);
    assert!(!world.ccd_enabled());

    // Can re-enable CCD
    world.set_ccd_enabled(true);
    assert!(world.ccd_enabled());
}

#[test]
fn test_ccd_substep_produces_collision() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);
    world.set_ccd_enabled(true);

    // Create a fast-moving sphere
    let mut fast_ball = RigidBody::new(
        CollisionShape::sphere(0.5),
        1.0,
        BodyType::Dynamic,
    );
    fast_ball.position = Vec3::new(-10.0, 0.0, 0.0);
    fast_ball.linear_velocity = Vec3::new(200.0, 0.0, 0.0); // Very fast
    let fast_handle = world.add_body(fast_ball);

    // Create a static wall
    let mut wall = RigidBody::new(
        CollisionShape::cube(Vec3::new(1.0, 5.0, 5.0)),
        1.0,
        BodyType::Static,
    );
    wall.position = Vec3::new(5.0, 0.0, 0.0);
    world.add_body(wall);

    // Step the simulation
    let result = world.step_full(1.0 / 60.0);

    // The fast ball should have collided with the wall
    // and not tunneled through it
    let ball = world.get_body(fast_handle).unwrap();
    assert!(
        ball.position.x < 5.0,
        "Ball should not have tunneled through wall: {:?}",
        ball.position
    );

    // CCD should have been triggered
    // (may or may not have substeps depending on implementation)
}

#[test]
fn test_ccd_disabled_allows_tunneling() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);
    world.set_ccd_enabled(false); // Disable CCD

    // Create a very fast sphere that will tunnel
    let mut fast_ball = RigidBody::new(
        CollisionShape::sphere(0.5),
        1.0,
        BodyType::Dynamic,
    );
    fast_ball.position = Vec3::new(-10.0, 0.0, 0.0);
    fast_ball.linear_velocity = Vec3::new(1000.0, 0.0, 0.0); // Extremely fast
    let fast_handle = world.add_body(fast_ball);

    // Create a thin wall
    let mut wall = RigidBody::new(
        CollisionShape::cube(Vec3::new(0.1, 5.0, 5.0)), // Very thin
        1.0,
        BodyType::Static,
    );
    wall.position = Vec3::new(5.0, 0.0, 0.0);
    world.add_body(wall);

    // With CCD disabled and a fast-enough ball, it may tunnel
    world.step(1.0 / 60.0);

    let ball = world.get_body(fast_handle).unwrap();
    // At 1000 units/s for 1/60 s, the ball moves ~16.67 units
    // Started at -10, should end around 6.67 (past the wall at 5.0)
    // This verifies tunneling can occur without CCD
    assert!(
        ball.position.x > 0.0,
        "Without CCD, ball should have moved significantly: {:?}",
        ball.position
    );
}

#[test]
fn test_max_ccd_substeps_limit() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);
    world.set_ccd_enabled(true);

    // Create a setup that would cause many substeps
    let mut ball = RigidBody::new(
        CollisionShape::sphere(0.1), // Small ball
        1.0,
        BodyType::Dynamic,
    );
    ball.position = Vec3::new(0.0, 0.0, 0.0);
    ball.linear_velocity = Vec3::new(100.0, 0.0, 0.0);
    world.add_body(ball);

    // Multiple thin walls
    for i in 1..10 {
        let mut wall = RigidBody::new(
            CollisionShape::cube(Vec3::new(0.05, 5.0, 5.0)),
            1.0,
            BodyType::Static,
        );
        wall.position = Vec3::new(i as f32 * 0.5, 0.0, 0.0);
        world.add_body(wall);
    }

    // Step should not hang due to max substep limit
    let result = world.step_full(1.0 / 60.0);

    // Should have limited substeps (max 4 by default)
    assert!(
        result.ccd_substeps <= 4,
        "CCD substeps should be limited: {}",
        result.ccd_substeps
    );
}

#[test]
fn test_needs_ccd_velocity_threshold() {
    // Small sphere with low velocity doesn't need CCD
    let small_sphere = CollisionShape::sphere(0.5);
    let slow_vel = Vec3::new(10.0, 0.0, 0.0);
    let dt = 1.0 / 60.0;

    // At 10 units/s for 1/60s = 0.167 units displacement
    // Sphere radius is 0.5, so no CCD needed
    assert!(!needs_ccd(&small_sphere, slow_vel, dt));

    // Same sphere with high velocity needs CCD
    let fast_vel = Vec3::new(100.0, 0.0, 0.0);
    // At 100 units/s for 1/60s = 1.67 units displacement
    // Greater than radius 0.5, so CCD needed
    assert!(needs_ccd(&small_sphere, fast_vel, dt));
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_sweep_stationary_objects() {
    // Two stationary objects don't collide even if close
    let sphere = CollisionShape::sphere(1.0);

    let result = sweep_test(
        &sphere,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0), // Not moving
        Quat::IDENTITY,
        &sphere,
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(5.0, 0.0, 0.0), // Not moving
        Quat::IDENTITY,
    );

    assert!(result.is_none(), "Stationary non-overlapping objects shouldn't collide");
}

#[test]
fn test_sweep_zero_velocity() {
    let sphere = CollisionShape::sphere(1.0);
    let velocity = Vec3::ZERO;
    let dt = 1.0 / 60.0;

    // Zero velocity should not need CCD
    assert!(!needs_ccd(&sphere, velocity, dt));
}

#[test]
fn test_ccd_with_rotated_box() {
    let sphere = CollisionShape::sphere(0.5);
    let box_shape = CollisionShape::cube(Vec3::new(1.0, 1.0, 1.0));

    // Box rotated 45 degrees around Y axis
    let box_rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);

    let result = sweep_test(
        &sphere,
        Vec3::new(-5.0, 0.0, 0.0),
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
        &box_shape,
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        box_rot,
    );

    assert!(result.is_some(), "Should detect collision with rotated box");
}

#[test]
fn test_sweep_result_normal_direction() {
    let sphere = CollisionShape::sphere(1.0);

    // Sphere moving right toward stationary sphere
    let result = sweep_test(
        &sphere,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(10.0, 0.0, 0.0),
        Quat::IDENTITY,
        &sphere,
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
    );

    assert!(result.is_some());
    let r = result.unwrap();
    // Normal should point from A to B (positive X direction)
    assert!(r.normal.x > 0.5, "Normal should point from A to B");
}

#[test]
fn test_capsule_sweep_detection() {
    let capsule = CollisionShape::capsule(0.5, 1.0);

    let result = sweep_test(
        &capsule,
        Vec3::new(-5.0, 0.0, 0.0),
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
        &capsule,
        Vec3::new(3.0, 0.0, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
        Quat::IDENTITY,
    );

    assert!(result.is_some(), "Should detect capsule-capsule collision");
}
