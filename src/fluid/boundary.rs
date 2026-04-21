//! Boundary conditions for SPH fluid simulation.
//!
//! This module provides boundary conditions that constrain fluid particles:
//!
//! - `PlaneBoundary`: Infinite plane boundary with penalty forces
//! - `BoxBoundary`: Axis-aligned box boundary composed of 6 plane boundaries
//! - `RigidBodyBoundary`: Dynamic boundary from a rigid body's collision shape
//! - `Boundary`: Trait for implementing custom boundary types

use glam::{Quat, Vec3};

use super::particle::FluidParticle;
use crate::collision::CollisionShape;

/// Trait for boundary conditions that apply forces to fluid particles.
///
/// Boundaries use penalty-based methods to keep particles inside the simulation domain.
/// When a particle penetrates the boundary, a repulsive force pushes it back.
pub trait Boundary {
    /// Compute the penalty force to apply to a particle.
    ///
    /// Returns `Vec3::ZERO` if the particle is not penetrating the boundary.
    fn apply_force(&self, particle: &FluidParticle) -> Vec3;
}

/// Infinite plane boundary.
///
/// The plane is defined by a point and a unit normal vector pointing inward
/// (toward the valid region). Particles on the wrong side of the plane receive
/// a penalty force proportional to their penetration depth.
#[derive(Debug, Clone, Copy)]
pub struct PlaneBoundary {
    /// Unit normal vector pointing inward (toward valid region)
    pub normal: Vec3,
    /// Any point on the plane
    pub point: Vec3,
    /// Penalty force coefficient (force per unit penetration)
    pub stiffness: f64,
    /// Velocity damping coefficient at the boundary
    pub damping: f64,
}

impl PlaneBoundary {
    /// Create a new plane boundary.
    ///
    /// # Arguments
    /// * `normal` - Unit normal vector pointing inward (toward valid region)
    /// * `point` - Any point on the plane
    ///
    /// # Panics
    /// Panics if `normal` is zero-length.
    pub fn new(normal: Vec3, point: Vec3) -> Self {
        let len = normal.length();
        assert!(len > 1e-10, "Normal vector must be non-zero");
        Self {
            normal: normal / len,
            point,
            stiffness: 10000.0,
            damping: 100.0,
        }
    }

    /// Create a plane boundary with custom stiffness and damping.
    pub fn with_params(normal: Vec3, point: Vec3, stiffness: f64, damping: f64) -> Self {
        let len = normal.length();
        assert!(len > 1e-10, "Normal vector must be non-zero");
        Self {
            normal: normal / len,
            point,
            stiffness,
            damping,
        }
    }

    /// Compute the signed distance from a point to the plane.
    ///
    /// Positive values indicate the point is on the valid (inward) side.
    /// Negative values indicate penetration.
    #[inline]
    pub fn signed_distance(&self, position: Vec3) -> f64 {
        (position - self.point).dot(self.normal) as f64
    }

    /// Compute the penalty force for a particle penetrating the plane.
    ///
    /// The force consists of:
    /// - A stiffness term proportional to penetration depth
    /// - A damping term proportional to velocity into the boundary
    pub fn apply_force(&self, particle: &FluidParticle) -> Vec3 {
        let distance = self.signed_distance(particle.position);

        // No force if particle is on the valid side
        if distance >= 0.0 {
            return Vec3::ZERO;
        }

        let penetration_depth = -distance;

        // Stiffness force: pushes particle out
        let stiffness_force = self.stiffness * penetration_depth;

        // Damping force: resists velocity into boundary
        let velocity_into_boundary = particle.velocity.dot(self.normal) as f64;
        let damping_force = if velocity_into_boundary < 0.0 {
            -self.damping * velocity_into_boundary
        } else {
            0.0
        };

        self.normal * (stiffness_force + damping_force) as f32
    }
}

impl Boundary for PlaneBoundary {
    fn apply_force(&self, particle: &FluidParticle) -> Vec3 {
        PlaneBoundary::apply_force(self, particle)
    }
}

/// Axis-aligned box boundary.
///
/// Constrains particles to stay within a rectangular box defined by its
/// minimum and maximum corner coordinates. Each face of the box acts as
/// a plane boundary.
#[derive(Debug, Clone, Copy)]
pub struct BoxBoundary {
    /// Corner with smallest coordinates
    pub min: Vec3,
    /// Corner with largest coordinates
    pub max: Vec3,
    /// Penalty force coefficient
    pub stiffness: f64,
    /// Velocity damping coefficient
    pub damping: f64,
}

impl BoxBoundary {
    /// Create a new box boundary.
    ///
    /// # Arguments
    /// * `min` - Corner with smallest coordinates
    /// * `max` - Corner with largest coordinates
    ///
    /// # Panics
    /// Panics if any component of `min` is greater than the corresponding component of `max`.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        assert!(
            min.x <= max.x && min.y <= max.y && min.z <= max.z,
            "min must be <= max in all components"
        );
        Self {
            min,
            max,
            stiffness: 10000.0,
            damping: 100.0,
        }
    }

    /// Create a box boundary with custom stiffness and damping.
    pub fn with_params(min: Vec3, max: Vec3, stiffness: f64, damping: f64) -> Self {
        assert!(
            min.x <= max.x && min.y <= max.y && min.z <= max.z,
            "min must be <= max in all components"
        );
        Self {
            min,
            max,
            stiffness,
            damping,
        }
    }

    /// Check if a point is inside the box.
    #[inline]
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Get the 6 plane boundaries that form this box.
    pub fn planes(&self) -> [PlaneBoundary; 6] {
        [
            // -X face (normal points +X)
            PlaneBoundary::with_params(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(self.min.x, 0.0, 0.0),
                self.stiffness,
                self.damping,
            ),
            // +X face (normal points -X)
            PlaneBoundary::with_params(
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(self.max.x, 0.0, 0.0),
                self.stiffness,
                self.damping,
            ),
            // -Y face (normal points +Y)
            PlaneBoundary::with_params(
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, self.min.y, 0.0),
                self.stiffness,
                self.damping,
            ),
            // +Y face (normal points -Y)
            PlaneBoundary::with_params(
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(0.0, self.max.y, 0.0),
                self.stiffness,
                self.damping,
            ),
            // -Z face (normal points +Z)
            PlaneBoundary::with_params(
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, self.min.z),
                self.stiffness,
                self.damping,
            ),
            // +Z face (normal points -Z)
            PlaneBoundary::with_params(
                Vec3::new(0.0, 0.0, -1.0),
                Vec3::new(0.0, 0.0, self.max.z),
                self.stiffness,
                self.damping,
            ),
        ]
    }

    /// Compute the sum of penalty forces from all 6 faces.
    pub fn apply_force(&self, particle: &FluidParticle) -> Vec3 {
        let mut total_force = Vec3::ZERO;
        for plane in &self.planes() {
            total_force += plane.apply_force(particle);
        }
        total_force
    }
}

impl Boundary for BoxBoundary {
    fn apply_force(&self, particle: &FluidParticle) -> Vec3 {
        BoxBoundary::apply_force(self, particle)
    }
}

/// Boundary from a rigid body's collision shape.
///
/// Treats a rigid body as a dynamic boundary for fluid particles.
/// The boundary updates each step based on the body's position and rotation.
/// Supports sphere, box, and capsule collision shapes.
#[derive(Debug, Clone)]
pub struct RigidBodyBoundary {
    /// Collision shape of the rigid body
    pub shape: CollisionShape,
    /// Current position of the rigid body
    pub position: Vec3,
    /// Current rotation of the rigid body
    pub rotation: Quat,
    /// Linear velocity of the rigid body (for relative damping)
    pub velocity: Vec3,
    /// Penalty force coefficient (force per unit penetration)
    pub stiffness: f64,
    /// Velocity damping coefficient at the boundary
    pub damping: f64,
}

impl RigidBodyBoundary {
    /// Create a new rigid body boundary.
    ///
    /// # Arguments
    /// * `shape` - The collision shape of the rigid body
    /// * `position` - Initial position
    /// * `rotation` - Initial rotation
    pub fn new(shape: CollisionShape, position: Vec3, rotation: Quat) -> Self {
        Self {
            shape,
            position,
            rotation,
            velocity: Vec3::ZERO,
            stiffness: 50000.0,
            damping: 0.5,
        }
    }

    /// Create a rigid body boundary with custom parameters.
    pub fn with_params(
        shape: CollisionShape,
        position: Vec3,
        rotation: Quat,
        stiffness: f64,
        damping: f64,
    ) -> Self {
        Self {
            shape,
            position,
            rotation,
            velocity: Vec3::ZERO,
            stiffness,
            damping,
        }
    }

    /// Update the boundary's transform and velocity.
    pub fn update(&mut self, position: Vec3, rotation: Quat, velocity: Vec3) {
        self.position = position;
        self.rotation = rotation;
        self.velocity = velocity;
    }

    /// Transform a world-space point to local space of the rigid body.
    #[inline]
    fn world_to_local(&self, world_point: Vec3) -> Vec3 {
        let inv_rotation = self.rotation.inverse();
        inv_rotation * (world_point - self.position)
    }

    /// Transform a local-space direction to world space.
    #[inline]
    fn local_to_world_dir(&self, local_dir: Vec3) -> Vec3 {
        self.rotation * local_dir
    }

    /// Compute signed distance and outward normal for a sphere shape.
    /// Returns (signed_distance, normal) where negative distance means penetration.
    fn sphere_sdf(&self, particle_pos: Vec3, radius: f32) -> (f64, Vec3) {
        let to_particle = particle_pos - self.position;
        let dist_to_center = to_particle.length();

        if dist_to_center < 1e-10 {
            // Particle at center of sphere - push in arbitrary direction
            return (-(radius as f64), Vec3::Y);
        }

        let normal = to_particle / dist_to_center;
        let signed_distance = (dist_to_center - radius) as f64;
        (signed_distance, normal)
    }

    /// Compute signed distance and outward normal for a box shape.
    /// Returns (signed_distance, normal) where negative distance means penetration.
    fn box_sdf(&self, particle_pos: Vec3, half_extents: Vec3) -> (f64, Vec3) {
        // Transform particle to local space
        let local_pos = self.world_to_local(particle_pos);

        // Compute distance to box surface in local space
        // For points inside: negative distance to nearest face
        // For points outside: positive distance to nearest point on surface

        let abs_pos = local_pos.abs();
        let diff = abs_pos - half_extents;

        // Check if inside the box
        if diff.x < 0.0 && diff.y < 0.0 && diff.z < 0.0 {
            // Inside - find closest face
            let penetrations = [
                (half_extents.x - abs_pos.x, Vec3::new(local_pos.x.signum(), 0.0, 0.0)),
                (half_extents.y - abs_pos.y, Vec3::new(0.0, local_pos.y.signum(), 0.0)),
                (half_extents.z - abs_pos.z, Vec3::new(0.0, 0.0, local_pos.z.signum())),
            ];

            let (min_pen, local_normal) = penetrations
                .iter()
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap();

            let world_normal = self.local_to_world_dir(*local_normal);
            return (-(*min_pen as f64), world_normal);
        }

        // Outside - compute distance to nearest point on surface
        let clamped = local_pos.clamp(-half_extents, half_extents);
        let to_surface = local_pos - clamped;
        let dist = to_surface.length();

        if dist < 1e-10 {
            // On surface - use largest component for normal
            let abs_diff = diff.abs();
            let local_normal = if abs_diff.x >= abs_diff.y && abs_diff.x >= abs_diff.z {
                Vec3::new(local_pos.x.signum(), 0.0, 0.0)
            } else if abs_diff.y >= abs_diff.z {
                Vec3::new(0.0, local_pos.y.signum(), 0.0)
            } else {
                Vec3::new(0.0, 0.0, local_pos.z.signum())
            };
            return (0.0, self.local_to_world_dir(local_normal));
        }

        let local_normal = to_surface / dist;
        let world_normal = self.local_to_world_dir(local_normal);
        (dist as f64, world_normal)
    }

    /// Compute signed distance and outward normal for a capsule shape.
    /// Capsule is oriented along Y axis with hemispherical caps.
    /// Returns (signed_distance, normal) where negative distance means penetration.
    fn capsule_sdf(&self, particle_pos: Vec3, radius: f32, half_height: f32) -> (f64, Vec3) {
        // Transform particle to local space
        let local_pos = self.world_to_local(particle_pos);

        // Capsule segment goes from (0, -half_height, 0) to (0, half_height, 0)
        // Find closest point on segment
        let segment_y = local_pos.y.clamp(-half_height, half_height);
        let closest_on_segment = Vec3::new(0.0, segment_y, 0.0);

        let to_particle = local_pos - closest_on_segment;
        let dist_to_axis = to_particle.length();

        if dist_to_axis < 1e-10 {
            // On the axis - push outward in X direction
            let local_normal = Vec3::X;
            let world_normal = self.local_to_world_dir(local_normal);
            return (-(radius as f64), world_normal);
        }

        let local_normal = to_particle / dist_to_axis;
        let world_normal = self.local_to_world_dir(local_normal);
        let signed_distance = (dist_to_axis - radius) as f64;
        (signed_distance, world_normal)
    }

    /// Compute penalty force for a particle near/inside this boundary.
    pub fn apply_force(&self, particle: &FluidParticle) -> Vec3 {
        let (signed_distance, normal) = match &self.shape {
            CollisionShape::Sphere { radius } => {
                self.sphere_sdf(particle.position, *radius)
            }
            CollisionShape::Box { half_extents } => {
                self.box_sdf(particle.position, *half_extents)
            }
            CollisionShape::Capsule { radius, half_height } => {
                self.capsule_sdf(particle.position, *radius, *half_height)
            }
            // For convex/mesh shapes, use AABB approximation
            CollisionShape::Convex { .. } | CollisionShape::Mesh { .. } => {
                let aabb = self.shape.local_aabb();
                self.box_sdf(particle.position, (aabb.max - aabb.min) * 0.5)
            }
        };

        // No force if particle is outside the boundary
        if signed_distance >= 0.0 {
            return Vec3::ZERO;
        }

        let penetration_depth = -signed_distance;

        // Stiffness force: pushes particle out
        let stiffness_force = self.stiffness * penetration_depth;

        // Relative velocity (particle velocity relative to body surface)
        let relative_velocity = particle.velocity - self.velocity;

        // Damping force: resists relative velocity into boundary
        let velocity_into_boundary = relative_velocity.dot(normal) as f64;
        let damping_force = if velocity_into_boundary < 0.0 {
            -self.damping * velocity_into_boundary * penetration_depth
        } else {
            0.0
        };

        normal * (stiffness_force + damping_force) as f32
    }
}

impl Boundary for RigidBodyBoundary {
    fn apply_force(&self, particle: &FluidParticle) -> Vec3 {
        RigidBodyBoundary::apply_force(self, particle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    // ==================== PlaneBoundary Construction Tests ====================

    #[test]
    fn plane_new_normalizes_normal() {
        let plane = PlaneBoundary::new(Vec3::new(0.0, 2.0, 0.0), Vec3::ZERO);
        assert!((plane.normal.length() - 1.0).abs() < EPSILON);
        assert!((plane.normal - Vec3::new(0.0, 1.0, 0.0)).length() < EPSILON);
    }

    #[test]
    fn plane_new_with_default_params() {
        let plane = PlaneBoundary::new(Vec3::Y, Vec3::ZERO);
        assert!((plane.stiffness - 10000.0).abs() < 1e-10);
        assert!((plane.damping - 100.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Normal vector must be non-zero")]
    fn plane_panics_on_zero_normal() {
        PlaneBoundary::new(Vec3::ZERO, Vec3::ZERO);
    }

    #[test]
    fn plane_with_params_sets_values() {
        let plane = PlaneBoundary::with_params(Vec3::Y, Vec3::ZERO, 5000.0, 50.0);
        assert!((plane.stiffness - 5000.0).abs() < 1e-10);
        assert!((plane.damping - 50.0).abs() < 1e-10);
    }

    // ==================== PlaneBoundary Signed Distance Tests ====================

    #[test]
    fn plane_signed_distance_positive_on_valid_side() {
        // Floor plane at y=0, normal pointing up
        let plane = PlaneBoundary::new(Vec3::Y, Vec3::ZERO);

        // Point above plane
        let dist = plane.signed_distance(Vec3::new(0.0, 1.0, 0.0));
        assert!(dist > 0.0, "Point above plane should have positive distance");
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn plane_signed_distance_negative_on_invalid_side() {
        let plane = PlaneBoundary::new(Vec3::Y, Vec3::ZERO);

        // Point below plane
        let dist = plane.signed_distance(Vec3::new(0.0, -0.5, 0.0));
        assert!(dist < 0.0, "Point below plane should have negative distance");
        assert!((dist + 0.5).abs() < 1e-10);
    }

    #[test]
    fn plane_signed_distance_zero_on_plane() {
        let plane = PlaneBoundary::new(Vec3::Y, Vec3::new(0.0, 1.0, 0.0));

        let dist = plane.signed_distance(Vec3::new(5.0, 1.0, -3.0));
        assert!(dist.abs() < 1e-10, "Point on plane should have zero distance");
    }

    // ==================== PlaneBoundary Force Tests ====================

    #[test]
    fn plane_no_force_when_not_penetrating() {
        let plane = PlaneBoundary::new(Vec3::Y, Vec3::ZERO);
        let particle = FluidParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0);

        let force = plane.apply_force(&particle);
        assert_eq!(force, Vec3::ZERO);
    }

    #[test]
    fn plane_applies_force_when_penetrating() {
        let plane = PlaneBoundary::new(Vec3::Y, Vec3::ZERO);
        let particle = FluidParticle::new(Vec3::new(0.0, -0.1, 0.0), 1.0);

        let force = plane.apply_force(&particle);

        // Force should point in normal direction (+Y)
        assert!(force.y > 0.0, "Force should push particle up");
        assert!(force.x.abs() < EPSILON);
        assert!(force.z.abs() < EPSILON);
    }

    #[test]
    fn plane_force_proportional_to_penetration() {
        let plane = PlaneBoundary::with_params(Vec3::Y, Vec3::ZERO, 1000.0, 0.0);

        let p1 = FluidParticle::new(Vec3::new(0.0, -0.1, 0.0), 1.0);
        let p2 = FluidParticle::new(Vec3::new(0.0, -0.2, 0.0), 1.0);

        let f1 = plane.apply_force(&p1);
        let f2 = plane.apply_force(&p2);

        // Double penetration = double force (with zero damping)
        assert!((f2.y / f1.y - 2.0).abs() < 0.01);
    }

    #[test]
    fn plane_damping_resists_velocity_into_boundary() {
        let plane = PlaneBoundary::with_params(Vec3::Y, Vec3::ZERO, 0.0, 100.0);

        // Particle penetrating and moving into boundary
        let mut particle = FluidParticle::new(Vec3::new(0.0, -0.1, 0.0), 1.0);
        particle.velocity = Vec3::new(0.0, -1.0, 0.0);

        let force = plane.apply_force(&particle);

        // Damping should resist downward velocity
        assert!(force.y > 0.0, "Damping should resist velocity into boundary");
    }

    #[test]
    fn plane_no_damping_for_velocity_away_from_boundary() {
        let plane = PlaneBoundary::with_params(Vec3::Y, Vec3::ZERO, 0.0, 100.0);

        // Particle penetrating but moving away from boundary
        let mut particle = FluidParticle::new(Vec3::new(0.0, -0.1, 0.0), 1.0);
        particle.velocity = Vec3::new(0.0, 1.0, 0.0);

        let force = plane.apply_force(&particle);

        // No damping force (only stiffness, which is 0)
        assert!(force.length() < EPSILON);
    }

    // ==================== BoxBoundary Construction Tests ====================

    #[test]
    fn box_new_creates_valid_boundary() {
        let bbox = BoxBoundary::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!((bbox.min - Vec3::new(-1.0, -1.0, -1.0)).length() < EPSILON);
        assert!((bbox.max - Vec3::new(1.0, 1.0, 1.0)).length() < EPSILON);
    }

    #[test]
    fn box_new_with_default_params() {
        let bbox = BoxBoundary::new(Vec3::ZERO, Vec3::ONE);
        assert!((bbox.stiffness - 10000.0).abs() < 1e-10);
        assert!((bbox.damping - 100.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "min must be <= max")]
    fn box_panics_on_invalid_bounds() {
        BoxBoundary::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 1.0));
    }

    // ==================== BoxBoundary Contains Tests ====================

    #[test]
    fn box_contains_point_inside() {
        let bbox = BoxBoundary::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(bbox.contains(Vec3::ZERO));
        assert!(bbox.contains(Vec3::new(0.5, 0.5, 0.5)));
        assert!(bbox.contains(Vec3::new(-0.9, -0.9, -0.9)));
    }

    #[test]
    fn box_contains_point_on_boundary() {
        let bbox = BoxBoundary::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(bbox.contains(Vec3::new(1.0, 0.0, 0.0)));
        assert!(bbox.contains(Vec3::new(-1.0, -1.0, -1.0)));
        assert!(bbox.contains(Vec3::new(1.0, 1.0, 1.0)));
    }

    #[test]
    fn box_does_not_contain_point_outside() {
        let bbox = BoxBoundary::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(!bbox.contains(Vec3::new(1.5, 0.0, 0.0)));
        assert!(!bbox.contains(Vec3::new(0.0, -1.5, 0.0)));
        assert!(!bbox.contains(Vec3::new(0.0, 0.0, 2.0)));
    }

    // ==================== BoxBoundary Force Tests ====================

    #[test]
    fn box_no_force_when_inside() {
        let bbox = BoxBoundary::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let particle = FluidParticle::new(Vec3::ZERO, 1.0);

        let force = bbox.apply_force(&particle);
        assert!(force.length() < EPSILON);
    }

    #[test]
    fn box_applies_force_when_penetrating_min_x() {
        let bbox = BoxBoundary::new(Vec3::new(0.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let particle = FluidParticle::new(Vec3::new(-0.1, 0.0, 0.0), 1.0);

        let force = bbox.apply_force(&particle);

        // Force should push particle toward +X
        assert!(force.x > 0.0, "Force should push particle inside");
    }

    #[test]
    fn box_applies_force_when_penetrating_max_y() {
        let bbox = BoxBoundary::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let particle = FluidParticle::new(Vec3::new(0.0, 1.1, 0.0), 1.0);

        let force = bbox.apply_force(&particle);

        // Force should push particle toward -Y
        assert!(force.y < 0.0, "Force should push particle inside");
    }

    #[test]
    fn box_applies_force_from_multiple_faces() {
        let bbox = BoxBoundary::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

        // Particle outside corner (penetrating two faces)
        let particle = FluidParticle::new(Vec3::new(-0.1, -0.1, 0.5), 1.0);

        let force = bbox.apply_force(&particle);

        // Should have forces from both -X and -Y faces
        assert!(force.x > 0.0, "Force x should push inside");
        assert!(force.y > 0.0, "Force y should push inside");
    }

    #[test]
    fn box_planes_returns_six_planes() {
        let bbox = BoxBoundary::new(Vec3::ZERO, Vec3::ONE);
        let planes = bbox.planes();
        assert_eq!(planes.len(), 6);
    }

    // ==================== Boundary Trait Tests ====================

    #[test]
    fn boundary_trait_works_for_plane() {
        let plane = PlaneBoundary::new(Vec3::Y, Vec3::ZERO);
        let particle = FluidParticle::new(Vec3::new(0.0, -0.1, 0.0), 1.0);

        let boundary: &dyn Boundary = &plane;
        let force = boundary.apply_force(&particle);

        assert!(force.y > 0.0);
    }

    #[test]
    fn boundary_trait_works_for_box() {
        let bbox = BoxBoundary::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let particle = FluidParticle::new(Vec3::new(-1.1, 0.0, 0.0), 1.0);

        let boundary: &dyn Boundary = &bbox;
        let force = boundary.apply_force(&particle);

        assert!(force.x > 0.0);
    }

    // ==================== Integration / Realistic Scenario Tests ====================

    #[test]
    fn floor_boundary_prevents_falling_through() {
        // Simulate a particle falling toward a floor
        let floor = PlaneBoundary::with_params(Vec3::Y, Vec3::ZERO, 10000.0, 500.0);

        let mut particle = FluidParticle::new(Vec3::new(0.0, 0.05, 0.0), 1.0);
        particle.velocity = Vec3::new(0.0, -2.0, 0.0);

        let dt = 0.001;

        // Simulate several steps
        for _ in 0..100 {
            let boundary_force = floor.apply_force(&particle);
            let gravity = Vec3::new(0.0, -9.81, 0.0);
            let acceleration = boundary_force / particle.mass as f32 + gravity;
            particle.velocity += acceleration * dt;
            particle.position += particle.velocity * dt;
        }

        // Particle should not have fallen far below the floor
        assert!(
            particle.position.y > -0.5,
            "Particle fell too far: y = {}",
            particle.position.y
        );
    }

    #[test]
    fn box_boundary_contains_bouncing_particle() {
        let bbox = BoxBoundary::with_params(
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            10000.0,
            200.0,
        );

        let mut particle = FluidParticle::new(Vec3::ZERO, 1.0);
        particle.velocity = Vec3::new(5.0, 3.0, -4.0);

        let dt = 0.0001;

        // Simulate many steps
        for _ in 0..10000 {
            let boundary_force = bbox.apply_force(&particle);
            let acceleration = boundary_force / particle.mass as f32;
            particle.velocity += acceleration * dt;
            particle.position += particle.velocity * dt;
        }

        // Particle should still be roughly within the box (may slightly penetrate)
        assert!(
            particle.position.x > -2.0 && particle.position.x < 2.0,
            "x out of range: {}",
            particle.position.x
        );
        assert!(
            particle.position.y > -2.0 && particle.position.y < 2.0,
            "y out of range: {}",
            particle.position.y
        );
        assert!(
            particle.position.z > -2.0 && particle.position.z < 2.0,
            "z out of range: {}",
            particle.position.z
        );
    }

    #[test]
    fn diagonal_plane_applies_correct_force() {
        // 45-degree plane
        let normal = Vec3::new(1.0, 1.0, 0.0).normalize();
        let plane = PlaneBoundary::with_params(normal, Vec3::ZERO, 1000.0, 0.0);

        // Point on the wrong side
        let particle = FluidParticle::new(Vec3::new(-0.5, -0.5, 0.0), 1.0);

        let force = plane.apply_force(&particle);

        // Force should be along the normal direction
        let force_normalized = force.normalize();
        let dot = force_normalized.dot(normal);
        assert!(
            (dot - 1.0).abs() < 0.01,
            "Force should be along normal, dot = {}",
            dot
        );
    }

    // ==================== RigidBodyBoundary Tests ====================

    #[test]
    fn rigid_body_boundary_sphere_no_force_outside() {
        let boundary = RigidBodyBoundary::new(
            CollisionShape::Sphere { radius: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        let particle = FluidParticle::new(Vec3::new(2.0, 0.0, 0.0), 1.0);

        let force = boundary.apply_force(&particle);
        assert!(force.length() < EPSILON, "Should be no force outside sphere");
    }

    #[test]
    fn rigid_body_boundary_sphere_force_inside() {
        let boundary = RigidBodyBoundary::new(
            CollisionShape::Sphere { radius: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        let particle = FluidParticle::new(Vec3::new(0.5, 0.0, 0.0), 1.0);

        let force = boundary.apply_force(&particle);
        assert!(force.x > 0.0, "Force should push particle outward (+X)");
    }

    #[test]
    fn rigid_body_boundary_sphere_force_proportional_to_penetration() {
        let boundary = RigidBodyBoundary::with_params(
            CollisionShape::Sphere { radius: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
            1000.0,
            0.0,
        );

        let p1 = FluidParticle::new(Vec3::new(0.9, 0.0, 0.0), 1.0); // 0.1 penetration
        let p2 = FluidParticle::new(Vec3::new(0.8, 0.0, 0.0), 1.0); // 0.2 penetration

        let f1 = boundary.apply_force(&p1);
        let f2 = boundary.apply_force(&p2);

        // Double penetration should give roughly double force
        assert!((f2.length() / f1.length() - 2.0).abs() < 0.1);
    }

    #[test]
    fn rigid_body_boundary_box_no_force_outside() {
        let boundary = RigidBodyBoundary::new(
            CollisionShape::Box { half_extents: Vec3::ONE },
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        let particle = FluidParticle::new(Vec3::new(2.0, 0.0, 0.0), 1.0);

        let force = boundary.apply_force(&particle);
        assert!(force.length() < EPSILON, "Should be no force outside box");
    }

    #[test]
    fn rigid_body_boundary_box_force_inside() {
        let boundary = RigidBodyBoundary::new(
            CollisionShape::Box { half_extents: Vec3::ONE },
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        let particle = FluidParticle::new(Vec3::new(0.5, 0.0, 0.0), 1.0);

        let force = boundary.apply_force(&particle);
        // Should push outward toward nearest face (+X)
        assert!(force.x > 0.0, "Force should push particle outward");
    }

    #[test]
    fn rigid_body_boundary_box_rotated() {
        // Rotate box 45 degrees around Y axis
        let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let boundary = RigidBodyBoundary::new(
            CollisionShape::Box { half_extents: Vec3::new(1.0, 1.0, 1.0) },
            Vec3::ZERO,
            rotation,
        );

        // Particle inside the rotated box
        let particle = FluidParticle::new(Vec3::new(0.5, 0.0, 0.5), 1.0);
        let force = boundary.apply_force(&particle);

        // Should have non-zero force pushing outward
        assert!(force.length() > 0.0, "Should push particle out of rotated box");
    }

    #[test]
    fn rigid_body_boundary_capsule_no_force_outside() {
        let boundary = RigidBodyBoundary::new(
            CollisionShape::Capsule { radius: 0.5, half_height: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        let particle = FluidParticle::new(Vec3::new(2.0, 0.0, 0.0), 1.0);

        let force = boundary.apply_force(&particle);
        assert!(force.length() < EPSILON, "Should be no force outside capsule");
    }

    #[test]
    fn rigid_body_boundary_capsule_force_inside() {
        let boundary = RigidBodyBoundary::new(
            CollisionShape::Capsule { radius: 1.0, half_height: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        // Particle inside the cylindrical part
        let particle = FluidParticle::new(Vec3::new(0.5, 0.0, 0.0), 1.0);

        let force = boundary.apply_force(&particle);
        assert!(force.x > 0.0, "Force should push particle outward (+X)");
    }

    #[test]
    fn rigid_body_boundary_capsule_hemispherical_cap() {
        let boundary = RigidBodyBoundary::new(
            CollisionShape::Capsule { radius: 1.0, half_height: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        // Particle inside the top hemispherical cap
        let particle = FluidParticle::new(Vec3::new(0.0, 1.5, 0.0), 1.0);

        let force = boundary.apply_force(&particle);
        assert!(force.y > 0.0, "Force should push particle upward out of cap");
    }

    #[test]
    fn rigid_body_boundary_update_position() {
        let mut boundary = RigidBodyBoundary::new(
            CollisionShape::Sphere { radius: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
        );

        // Particle outside sphere at origin
        let particle = FluidParticle::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
        let force1 = boundary.apply_force(&particle);
        assert!(force1.length() < EPSILON, "No force initially");

        // Move sphere so particle is now inside
        boundary.update(Vec3::new(1.5, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
        let force2 = boundary.apply_force(&particle);
        assert!(force2.length() > 0.0, "Force after sphere moved to particle");
    }

    #[test]
    fn rigid_body_boundary_damping_with_body_velocity() {
        let mut boundary = RigidBodyBoundary::with_params(
            CollisionShape::Sphere { radius: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
            1000.0,
            10.0,
        );

        // Particle inside, stationary
        let mut particle = FluidParticle::new(Vec3::new(0.5, 0.0, 0.0), 1.0);
        particle.velocity = Vec3::ZERO;

        // Body moving toward particle - relative velocity is into boundary
        boundary.velocity = Vec3::new(1.0, 0.0, 0.0);
        let force = boundary.apply_force(&particle);

        // Damping should add to the force since particle is "moving into" body
        assert!(force.x > 0.0, "Damping should contribute to pushing particle out");
    }

    #[test]
    fn rigid_body_boundary_trait_implementation() {
        let boundary = RigidBodyBoundary::new(
            CollisionShape::Sphere { radius: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        let particle = FluidParticle::new(Vec3::new(0.5, 0.0, 0.0), 1.0);

        // Test through trait
        let boundary_ref: &dyn Boundary = &boundary;
        let force = boundary_ref.apply_force(&particle);
        assert!(force.x > 0.0);
    }
}
