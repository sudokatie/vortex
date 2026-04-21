//! Continuous Collision Detection (CCD) module.
//!
//! This module provides sweep tests and time-of-impact calculations to detect
//! collisions between fast-moving objects that might tunnel through each other
//! with discrete collision detection.

use glam::{Quat, Vec3};

use super::gjk::{gjk_intersection, gjk_distance};
use super::CollisionShape;
use crate::math::Transform;
use crate::world::BodyHandle;

/// Maximum iterations for conservative advancement.
const MAX_CA_ITERATIONS: usize = 32;

/// Convergence threshold for conservative advancement.
const CA_EPSILON: f32 = 1e-4;

/// Time of Impact result.
#[derive(Debug, Clone, Copy)]
pub struct TimeOfImpact {
    /// Time parameter in [0, 1] where 0 is start and 1 is end of the sweep.
    pub time: f32,
    /// Contact point in world space.
    pub point: Vec3,
    /// Contact normal (from shape_a to shape_b).
    pub normal: Vec3,
    /// Handle of the first shape involved.
    pub shape_a: BodyHandle,
    /// Handle of the second shape involved.
    pub shape_b: BodyHandle,
}

impl TimeOfImpact {
    /// Create a new time of impact result.
    pub fn new(
        time: f32,
        point: Vec3,
        normal: Vec3,
        shape_a: BodyHandle,
        shape_b: BodyHandle,
    ) -> Self {
        Self {
            time,
            point,
            normal,
            shape_a,
            shape_b,
        }
    }
}

/// Result of a sweep test.
#[derive(Debug, Clone, Copy)]
pub struct SweepResult {
    /// Time of impact in [0, 1] range.
    pub time: f32,
    /// Contact point at time of impact.
    pub point: Vec3,
    /// Contact normal at time of impact.
    pub normal: Vec3,
}

impl SweepResult {
    /// Create a new sweep result.
    pub fn new(time: f32, point: Vec3, normal: Vec3) -> Self {
        Self { time, point, normal }
    }
}

/// Check if a body needs CCD based on its velocity and shape.
///
/// A body needs CCD if it moves more than half its characteristic size per timestep.
pub fn needs_ccd(shape: &CollisionShape, velocity: Vec3, dt: f32) -> bool {
    let displacement = velocity.length() * dt;
    let threshold = shape_ccd_threshold(shape);
    displacement > threshold
}

/// Get the CCD threshold for a shape (half the smallest dimension).
fn shape_ccd_threshold(shape: &CollisionShape) -> f32 {
    match shape {
        CollisionShape::Sphere { radius } => *radius,
        CollisionShape::Box { half_extents } => {
            half_extents.x.min(half_extents.y).min(half_extents.z)
        }
        CollisionShape::Capsule { radius, .. } => *radius,
        CollisionShape::Convex { vertices, .. } => {
            if vertices.is_empty() {
                0.5
            } else {
                // Use smallest extent of AABB as threshold
                let mut min = vertices[0];
                let mut max = vertices[0];
                for v in vertices {
                    min = min.min(*v);
                    max = max.max(*v);
                }
                let extents = (max - min) * 0.5;
                extents.x.min(extents.y).min(extents.z)
            }
        }
    }
}

/// Perform a sweep test between two shapes moving along linear trajectories.
///
/// Returns the time of first contact in [0, 1] range, or None if no collision occurs.
pub fn sweep_test(
    shape_a: &CollisionShape,
    pos_a_start: Vec3,
    pos_a_end: Vec3,
    rot_a: Quat,
    shape_b: &CollisionShape,
    pos_b_start: Vec3,
    pos_b_end: Vec3,
    rot_b: Quat,
) -> Option<SweepResult> {
    // Check if already overlapping at start
    let transform_a_start = Transform::new(pos_a_start, rot_a);
    let transform_b_start = Transform::new(pos_b_start, rot_b);

    if gjk_intersection(shape_a, &transform_a_start, shape_b, &transform_b_start).is_some() {
        // Already overlapping - return time 0
        let contact_point = (pos_a_start + pos_b_start) * 0.5;
        let normal = (pos_b_start - pos_a_start).normalize_or_zero();
        return Some(SweepResult::new(0.0, contact_point, normal));
    }

    // Note: We could check transform_a_end and transform_b_end for additional
    // early-out optimizations, but this is handled by the sweep algorithms.

    // Use specialized sweep for sphere-sphere
    if let (CollisionShape::Sphere { radius: r_a }, CollisionShape::Sphere { radius: r_b }) =
        (shape_a, shape_b)
    {
        return sweep_sphere_sphere(
            pos_a_start, pos_a_end, *r_a,
            pos_b_start, pos_b_end, *r_b,
        );
    }

    // For general convex shapes, use conservative advancement
    conservative_advancement(
        shape_a,
        pos_a_start,
        pos_a_end,
        rot_a,
        shape_b,
        pos_b_start,
        pos_b_end,
        rot_b,
    )
}

/// Analytical sphere-sphere sweep test.
///
/// Solves the quadratic equation for when two moving spheres first touch.
fn sweep_sphere_sphere(
    pos_a_start: Vec3,
    pos_a_end: Vec3,
    radius_a: f32,
    pos_b_start: Vec3,
    pos_b_end: Vec3,
    radius_b: f32,
) -> Option<SweepResult> {
    // Relative motion: treat A as stationary and B as moving
    let vel_a = pos_a_end - pos_a_start;
    let vel_b = pos_b_end - pos_b_start;
    let rel_vel = vel_b - vel_a;

    let rel_pos = pos_b_start - pos_a_start;
    let combined_radius = radius_a + radius_b;

    // Quadratic equation: |rel_pos + t * rel_vel|^2 = combined_radius^2
    // a*t^2 + b*t + c = 0
    let a = rel_vel.dot(rel_vel);
    let b = 2.0 * rel_pos.dot(rel_vel);
    let c = rel_pos.dot(rel_pos) - combined_radius * combined_radius;

    // Check if already overlapping
    if c <= 0.0 {
        let contact_point = (pos_a_start + pos_b_start) * 0.5;
        let normal = rel_pos.normalize_or_zero();
        return Some(SweepResult::new(0.0, contact_point, normal));
    }

    // Check if moving apart or parallel
    if a.abs() < 1e-10 {
        return None; // No relative motion
    }

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None; // No intersection
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);

    // Find first valid intersection time
    let t = if (0.0..=1.0).contains(&t1) {
        t1
    } else if (0.0..=1.0).contains(&t2) {
        t2
    } else if t1 > 1.0 && t2 > 1.0 {
        return None; // Collision is in the future beyond this timestep
    } else if t1 < 0.0 && t2 < 0.0 {
        return None; // Collision was in the past
    } else {
        return None;
    };

    // Calculate contact point and normal at time t
    let pos_a_at_t = pos_a_start + vel_a * t;
    let pos_b_at_t = pos_b_start + vel_b * t;
    let normal = (pos_b_at_t - pos_a_at_t).normalize_or_zero();
    let contact_point = pos_a_at_t + normal * radius_a;

    Some(SweepResult::new(t, contact_point, normal))
}

/// Conservative advancement for general convex shape sweep test.
///
/// Iteratively advances time while maintaining a safe distance until shapes touch.
fn conservative_advancement(
    shape_a: &CollisionShape,
    pos_a_start: Vec3,
    pos_a_end: Vec3,
    rot_a: Quat,
    shape_b: &CollisionShape,
    pos_b_start: Vec3,
    pos_b_end: Vec3,
    rot_b: Quat,
) -> Option<SweepResult> {
    let vel_a = pos_a_end - pos_a_start;
    let vel_b = pos_b_end - pos_b_start;
    let rel_vel = vel_b - vel_a;
    let rel_speed = rel_vel.length();

    // If no relative motion, check static intersection
    if rel_speed < 1e-10 {
        let transform_a = Transform::new(pos_a_start, rot_a);
        let transform_b = Transform::new(pos_b_start, rot_b);
        if gjk_intersection(shape_a, &transform_a, shape_b, &transform_b).is_some() {
            let contact_point = (pos_a_start + pos_b_start) * 0.5;
            let normal = (pos_b_start - pos_a_start).normalize_or_zero();
            return Some(SweepResult::new(0.0, contact_point, normal));
        }
        return None;
    }

    let mut t = 0.0;

    for _ in 0..MAX_CA_ITERATIONS {
        // Interpolate positions at current time
        let pos_a_t = pos_a_start.lerp(pos_a_end, t);
        let pos_b_t = pos_b_start.lerp(pos_b_end, t);

        let transform_a = Transform::new(pos_a_t, rot_a);
        let transform_b = Transform::new(pos_b_t, rot_b);

        // Check for intersection
        if gjk_intersection(shape_a, &transform_a, shape_b, &transform_b).is_some() {
            // Found intersection - binary search for exact time
            return binary_search_toi(
                shape_a, pos_a_start, pos_a_end, rot_a,
                shape_b, pos_b_start, pos_b_end, rot_b,
                (t - 0.1).max(0.0), t,
            );
        }

        // Get distance between shapes
        let distance = gjk_distance(shape_a, &transform_a, shape_b, &transform_b);

        // Check for convergence
        if distance < CA_EPSILON {
            // Close enough - compute contact info
            let normal = (pos_b_t - pos_a_t).normalize_or_zero();
            let contact_point = pos_a_t + normal * (distance * 0.5);
            return Some(SweepResult::new(t, contact_point, normal));
        }

        // Conservative time advancement
        let dt = distance / rel_speed;
        t += dt * 0.9; // Factor < 1 for safety

        if t > 1.0 {
            return None; // No collision in this timestep
        }
    }

    None // Max iterations reached without finding collision
}

/// Binary search to find exact time of impact.
fn binary_search_toi(
    shape_a: &CollisionShape,
    pos_a_start: Vec3,
    pos_a_end: Vec3,
    rot_a: Quat,
    shape_b: &CollisionShape,
    pos_b_start: Vec3,
    pos_b_end: Vec3,
    rot_b: Quat,
    t_min: f32,
    t_max: f32,
) -> Option<SweepResult> {
    let mut lo = t_min;
    let mut hi = t_max;

    for _ in 0..16 {
        let mid = (lo + hi) * 0.5;

        let pos_a_mid = pos_a_start.lerp(pos_a_end, mid);
        let pos_b_mid = pos_b_start.lerp(pos_b_end, mid);

        let transform_a = Transform::new(pos_a_mid, rot_a);
        let transform_b = Transform::new(pos_b_mid, rot_b);

        if gjk_intersection(shape_a, &transform_a, shape_b, &transform_b).is_some() {
            hi = mid;
        } else {
            lo = mid;
        }

        if hi - lo < CA_EPSILON {
            break;
        }
    }

    // Return result at the found time
    let t = (lo + hi) * 0.5;
    let pos_a_t = pos_a_start.lerp(pos_a_end, t);
    let pos_b_t = pos_b_start.lerp(pos_b_end, t);
    let normal = (pos_b_t - pos_a_t).normalize_or_zero();
    let contact_point = (pos_a_t + pos_b_t) * 0.5;

    Some(SweepResult::new(t, contact_point, normal))
}

/// Calculate time of impact between two moving bodies.
///
/// Returns the earliest time of impact, or None if shapes never collide during
/// the timestep.
pub fn calculate_toi(
    handle_a: BodyHandle,
    shape_a: &CollisionShape,
    pos_a_start: Vec3,
    vel_a: Vec3,
    rot_a: Quat,
    handle_b: BodyHandle,
    shape_b: &CollisionShape,
    pos_b_start: Vec3,
    vel_b: Vec3,
    rot_b: Quat,
    dt: f32,
) -> Option<TimeOfImpact> {
    let pos_a_end = pos_a_start + vel_a * dt;
    let pos_b_end = pos_b_start + vel_b * dt;

    let result = sweep_test(
        shape_a, pos_a_start, pos_a_end, rot_a,
        shape_b, pos_b_start, pos_b_end, rot_b,
    )?;

    Some(TimeOfImpact::new(
        result.time,
        result.point,
        result.normal,
        handle_a,
        handle_b,
    ))
}

/// Perform sweep test for a sphere against a box.
pub fn sweep_sphere_box(
    sphere_pos_start: Vec3,
    sphere_pos_end: Vec3,
    sphere_radius: f32,
    box_pos_start: Vec3,
    box_pos_end: Vec3,
    box_half_extents: Vec3,
    box_rot: Quat,
) -> Option<SweepResult> {
    let sphere = CollisionShape::Sphere { radius: sphere_radius };
    let box_shape = CollisionShape::Box { half_extents: box_half_extents };

    sweep_test(
        &sphere, sphere_pos_start, sphere_pos_end, Quat::IDENTITY,
        &box_shape, box_pos_start, box_pos_end, box_rot,
    )
}

/// Perform sweep test for a sphere against a capsule.
pub fn sweep_sphere_capsule(
    sphere_pos_start: Vec3,
    sphere_pos_end: Vec3,
    sphere_radius: f32,
    capsule_pos_start: Vec3,
    capsule_pos_end: Vec3,
    capsule_radius: f32,
    capsule_half_height: f32,
    capsule_rot: Quat,
) -> Option<SweepResult> {
    let sphere = CollisionShape::Sphere { radius: sphere_radius };
    let capsule = CollisionShape::Capsule {
        radius: capsule_radius,
        half_height: capsule_half_height,
    };

    sweep_test(
        &sphere, sphere_pos_start, sphere_pos_end, Quat::IDENTITY,
        &capsule, capsule_pos_start, capsule_pos_end, capsule_rot,
    )
}

/// Perform sweep test for two boxes.
pub fn sweep_box_box(
    box_a_pos_start: Vec3,
    box_a_pos_end: Vec3,
    box_a_half_extents: Vec3,
    box_a_rot: Quat,
    box_b_pos_start: Vec3,
    box_b_pos_end: Vec3,
    box_b_half_extents: Vec3,
    box_b_rot: Quat,
) -> Option<SweepResult> {
    let box_a = CollisionShape::Box { half_extents: box_a_half_extents };
    let box_b = CollisionShape::Box { half_extents: box_b_half_extents };

    sweep_test(
        &box_a, box_a_pos_start, box_a_pos_end, box_a_rot,
        &box_b, box_b_pos_start, box_b_pos_end, box_b_rot,
    )
}

/// Perform sweep test for two capsules.
pub fn sweep_capsule_capsule(
    cap_a_pos_start: Vec3,
    cap_a_pos_end: Vec3,
    cap_a_radius: f32,
    cap_a_half_height: f32,
    cap_a_rot: Quat,
    cap_b_pos_start: Vec3,
    cap_b_pos_end: Vec3,
    cap_b_radius: f32,
    cap_b_half_height: f32,
    cap_b_rot: Quat,
) -> Option<SweepResult> {
    let cap_a = CollisionShape::Capsule {
        radius: cap_a_radius,
        half_height: cap_a_half_height,
    };
    let cap_b = CollisionShape::Capsule {
        radius: cap_b_radius,
        half_height: cap_b_half_height,
    };

    sweep_test(
        &cap_a, cap_a_pos_start, cap_a_pos_end, cap_a_rot,
        &cap_b, cap_b_pos_start, cap_b_pos_end, cap_b_rot,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sphere_sweep_hit() {
        // Two spheres moving toward each other
        // A: starts at -5, moves to 0 (velocity = +5)
        // B: starts at 5, moves to 0 (velocity = -5)
        // Relative velocity = -5 - 5 = -10 (B approaches A at speed 10)
        // Initial distance = 10, combined radius = 2
        // Collision when: 10 - 10*t = 2 => t = 0.8
        let result = sweep_sphere_sphere(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            1.0,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            1.0,
        );

        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.time >= 0.0 && r.time <= 1.0);
        // At t=0.8: A at -5 + 0.8*5 = -1, B at 5 - 0.8*5 = 1
        // Distance = 2 = sum of radii. Perfect contact.
        assert!((r.time - 0.8).abs() < 0.1);
    }

    #[test]
    fn test_sphere_sphere_sweep_miss() {
        // Two spheres moving in parallel
        let result = sweep_sphere_sphere(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            1.0,
            Vec3::new(0.0, 5.0, 0.0), // 5 units apart in Y
            Vec3::new(10.0, 5.0, 0.0),
            1.0,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_sphere_sphere_already_overlapping() {
        let result = sweep_sphere_sphere(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            2.0,
            Vec3::new(1.0, 0.0, 0.0), // Already overlapping
            Vec3::new(2.0, 0.0, 0.0),
            2.0,
        );

        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.time, 0.0);
    }

    #[test]
    fn test_sweep_test_box_sphere() {
        let sphere = CollisionShape::sphere(1.0);
        let box_shape = CollisionShape::cube(Vec3::ONE);

        // Sphere moves from -5 to 5, box is at x=2
        // Sphere reaches box when sphere_x + radius = box_x - half_extent
        // sphere_x + 1 = 2 - 1 = 1 => sphere_x = 0
        // Sphere travels from -5 to 5 (10 units), reaches x=0 at t = 5/10 = 0.5
        let result = sweep_test(
            &sphere,
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Quat::IDENTITY,
            &box_shape,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Quat::IDENTITY,
        );

        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.time >= 0.0 && r.time <= 1.0);
    }

    #[test]
    fn test_sweep_test_no_collision() {
        let sphere = CollisionShape::sphere(1.0);
        let box_shape = CollisionShape::cube(Vec3::ONE);

        let result = sweep_test(
            &sphere,
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(-3.0, 0.0, 0.0), // Doesn't reach the box
            Quat::IDENTITY,
            &box_shape,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Quat::IDENTITY,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_needs_ccd_fast_sphere() {
        let sphere = CollisionShape::sphere(1.0);
        let fast_velocity = Vec3::new(100.0, 0.0, 0.0);
        let dt = 1.0 / 60.0;

        assert!(needs_ccd(&sphere, fast_velocity, dt));
    }

    #[test]
    fn test_needs_ccd_slow_sphere() {
        let sphere = CollisionShape::sphere(1.0);
        let slow_velocity = Vec3::new(1.0, 0.0, 0.0);
        let dt = 1.0 / 60.0;

        assert!(!needs_ccd(&sphere, slow_velocity, dt));
    }

    #[test]
    fn test_needs_ccd_box() {
        let box_shape = CollisionShape::cube(Vec3::new(0.5, 0.5, 0.5));
        let fast_velocity = Vec3::new(50.0, 0.0, 0.0);
        let dt = 1.0 / 60.0;

        assert!(needs_ccd(&box_shape, fast_velocity, dt));
    }

    #[test]
    fn test_sweep_result_time_bounds() {
        let sphere = CollisionShape::sphere(0.5);

        // Head-on collision
        let result = sweep_test(
            &sphere,
            Vec3::new(-3.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Quat::IDENTITY,
            &sphere,
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(-3.0, 0.0, 0.0),
            Quat::IDENTITY,
        );

        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.time >= 0.0 && r.time <= 1.0);
    }

    #[test]
    fn test_capsule_sweep() {
        // Capsule A moves from -5 to 5 (10 units), capsule B is at x=2
        // Combined radius = 0.5 + 0.5 = 1.0
        // Collision when: capsule_a_x + 0.5 = capsule_b_x - 0.5
        // capsule_a_x = 2 - 1 = 1
        // Capsule travels from -5 to 5 (10 units), reaches x=1 at t = 6/10 = 0.6
        let result = sweep_capsule_capsule(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            0.5,
            1.0,
            Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            0.5,
            1.0,
            Quat::IDENTITY,
        );

        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.time >= 0.0 && r.time <= 1.0);
    }

    #[test]
    fn test_box_box_sweep() {
        // Box A moves from -5 to 5, box B is at x=2
        // Both boxes have half_extents = 0.5
        // Collision when: box_a_x + 0.5 = box_b_x - 0.5
        // box_a_x = 2 - 1 = 1
        // Box travels from -5 to 5 (10 units), reaches x=1 at t = 6/10 = 0.6
        let result = sweep_box_box(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Quat::IDENTITY,
        );

        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.time >= 0.0 && r.time <= 1.0);
    }
}
