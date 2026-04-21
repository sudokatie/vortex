//! Fluid-rigid body coupling for two-way interaction.
//!
//! This module provides forces that couple SPH fluid simulation with rigid body dynamics:
//!
//! - `compute_buoyancy`: Archimedes buoyancy force based on submerged volume
//! - `compute_drag`: Viscous drag force from fluid-body relative velocity
//! - `apply_fluid_forces_to_body`: Combined buoyancy and drag with torque
//! - `FluidCouplingParams`: Configuration parameters for coupling

use glam::{Quat, Vec3};

use super::particle::FluidParticle;
use super::boundary::RigidBodyBoundary;
use crate::collision::CollisionShape;

/// Parameters for fluid-rigid body coupling.
#[derive(Debug, Clone)]
pub struct FluidCouplingParams {
    /// Enable buoyancy force
    pub buoyancy_enabled: bool,
    /// Enable drag force
    pub drag_enabled: bool,
    /// Boundary stiffness for pushing fluid particles out of bodies
    pub stiffness: f64,
    /// Boundary damping coefficient
    pub damping: f64,
    /// Drag coefficient (dimensionless, typically 0.5-1.0 for blunt objects)
    pub drag_coefficient: f64,
    /// Rest density of the fluid (kg/m^3, default water = 1000)
    pub fluid_density: f64,
    /// Smoothing radius for neighbor queries
    pub smoothing_radius: f64,
}

impl Default for FluidCouplingParams {
    fn default() -> Self {
        Self {
            buoyancy_enabled: true,
            drag_enabled: true,
            stiffness: 50000.0,
            damping: 0.5,
            drag_coefficient: 0.8,
            fluid_density: 1000.0,
            smoothing_radius: 0.5,
        }
    }
}

impl FluidCouplingParams {
    /// Create new coupling parameters with all options enabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create coupling parameters with buoyancy only.
    pub fn buoyancy_only() -> Self {
        Self {
            buoyancy_enabled: true,
            drag_enabled: false,
            ..Default::default()
        }
    }

    /// Create coupling parameters with drag only.
    pub fn drag_only() -> Self {
        Self {
            buoyancy_enabled: false,
            drag_enabled: true,
            ..Default::default()
        }
    }

    /// Create disabled coupling parameters.
    pub fn disabled() -> Self {
        Self {
            buoyancy_enabled: false,
            drag_enabled: false,
            ..Default::default()
        }
    }

    /// Set the fluid density.
    pub fn with_fluid_density(mut self, density: f64) -> Self {
        self.fluid_density = density;
        self
    }

    /// Set the drag coefficient.
    pub fn with_drag_coefficient(mut self, coeff: f64) -> Self {
        self.drag_coefficient = coeff;
        self
    }

    /// Set the boundary stiffness.
    pub fn with_stiffness(mut self, stiffness: f64) -> Self {
        self.stiffness = stiffness;
        self
    }
}

/// Compute buoyancy force using Archimedes' principle.
///
/// Buoyancy = fluid_density * submerged_volume * gravity_magnitude * up_direction
///
/// # Arguments
/// * `fluid_density` - Density of the fluid near the body (kg/m^3)
/// * `submerged_volume` - Volume of the body submerged in fluid (m^3)
/// * `gravity` - Gravity vector (typically pointing down)
///
/// # Returns
/// Buoyancy force vector (pointing opposite to gravity)
pub fn compute_buoyancy(fluid_density: f64, submerged_volume: f64, gravity: Vec3) -> Vec3 {
    // F_buoyancy = rho * V * g (opposite to gravity direction)
    let gravity_magnitude = gravity.length() as f64;
    if gravity_magnitude < 1e-10 || submerged_volume < 1e-10 {
        return Vec3::ZERO;
    }

    let force_magnitude = fluid_density * submerged_volume * gravity_magnitude;
    let up_direction = -gravity.normalize();
    up_direction * force_magnitude as f32
}

/// Compute drag force from relative velocity.
///
/// Drag = 0.5 * drag_coeff * fluid_density * area * |v_rel|^2 * direction
///
/// # Arguments
/// * `relative_velocity` - Velocity of fluid relative to body (fluid_vel - body_vel)
/// * `drag_coefficient` - Dimensionless drag coefficient (typically 0.5-1.0)
/// * `fluid_density` - Density of the fluid (kg/m^3)
/// * `cross_section_area` - Cross-sectional area perpendicular to flow (m^2)
///
/// # Returns
/// Drag force vector (in direction of relative velocity)
pub fn compute_drag(
    relative_velocity: Vec3,
    drag_coefficient: f64,
    fluid_density: f64,
    cross_section_area: f64,
) -> Vec3 {
    let speed = relative_velocity.length() as f64;
    if speed < 1e-10 {
        return Vec3::ZERO;
    }

    // F_drag = 0.5 * Cd * rho * A * v^2
    let force_magnitude = 0.5 * drag_coefficient * fluid_density * cross_section_area * speed * speed;
    let direction = relative_velocity.normalize();
    direction * force_magnitude as f32
}

/// Estimate the volume of a collision shape.
pub fn estimate_volume(shape: &CollisionShape) -> f64 {
    match shape {
        CollisionShape::Sphere { radius } => {
            // V = (4/3) * pi * r^3
            (4.0 / 3.0) * std::f64::consts::PI * (*radius as f64).powi(3)
        }
        CollisionShape::Box { half_extents } => {
            // V = 8 * hx * hy * hz
            8.0 * half_extents.x as f64 * half_extents.y as f64 * half_extents.z as f64
        }
        CollisionShape::Capsule { radius, half_height } => {
            // V = pi * r^2 * (2 * half_height) + (4/3) * pi * r^3
            let r = *radius as f64;
            let h = (*half_height * 2.0) as f64;
            std::f64::consts::PI * r * r * h + (4.0 / 3.0) * std::f64::consts::PI * r.powi(3)
        }
        CollisionShape::Convex { .. } | CollisionShape::Mesh { .. } => {
            // Approximate using AABB
            let aabb = shape.local_aabb();
            let extents = aabb.max - aabb.min;
            (extents.x * extents.y * extents.z) as f64
        }
    }
}

/// Estimate the cross-sectional area of a shape (perpendicular to velocity).
pub fn estimate_cross_section(shape: &CollisionShape, velocity_dir: Vec3) -> f64 {
    let dir = velocity_dir.normalize_or_zero();
    if dir.length_squared() < 0.5 {
        // No velocity direction - use average
        return estimate_average_cross_section(shape);
    }

    match shape {
        CollisionShape::Sphere { radius } => {
            // Circle: A = pi * r^2
            std::f64::consts::PI * (*radius as f64).powi(2)
        }
        CollisionShape::Box { half_extents } => {
            // Project box onto plane perpendicular to velocity
            let abs_dir = dir.abs();
            let a = half_extents.x as f64 * 2.0;
            let b = half_extents.y as f64 * 2.0;
            let c = half_extents.z as f64 * 2.0;
            // Approximate: weighted sum of face areas
            abs_dir.x as f64 * b * c + abs_dir.y as f64 * a * c + abs_dir.z as f64 * a * b
        }
        CollisionShape::Capsule { radius, half_height } => {
            let r = *radius as f64;
            let h = (*half_height * 2.0) as f64;
            let abs_y = dir.y.abs() as f64;
            // Blend between circle (end-on) and rectangle (side-on)
            let circle_area = std::f64::consts::PI * r * r;
            let rect_area = 2.0 * r * (h + 2.0 * r);
            abs_y * circle_area + (1.0 - abs_y) * rect_area
        }
        CollisionShape::Convex { .. } | CollisionShape::Mesh { .. } => {
            estimate_average_cross_section(shape)
        }
    }
}

fn estimate_average_cross_section(shape: &CollisionShape) -> f64 {
    let aabb = shape.local_aabb();
    let extents = aabb.max - aabb.min;
    // Average of the three face areas
    let xy = (extents.x * extents.y) as f64;
    let xz = (extents.x * extents.z) as f64;
    let yz = (extents.y * extents.z) as f64;
    (xy + xz + yz) / 3.0
}

/// Count fluid particles near/inside a body to estimate submersion.
///
/// Returns (particle_count_inside, average_fluid_velocity, center_of_buoyancy)
pub fn count_particles_near_body(
    particles: &[FluidParticle],
    shape: &CollisionShape,
    position: Vec3,
    rotation: Quat,
    margin: f64,
) -> (usize, Vec3, Vec3) {
    let boundary = RigidBodyBoundary::with_params(
        shape.clone(),
        position,
        rotation,
        1.0, // stiffness doesn't matter for distance check
        0.0,
    );

    let mut count = 0;
    let mut velocity_sum = Vec3::ZERO;
    let mut position_sum = Vec3::ZERO;

    for particle in particles {
        // Check if particle is within margin of the body surface
        let force = boundary.apply_force(particle);
        if force.length_squared() > 0.0 {
            // Particle is inside the body
            count += 1;
            velocity_sum += particle.velocity;
            position_sum += particle.position;
        } else {
            // Check if particle is near the surface (within margin)
            let to_center = position - particle.position;
            let dist_to_center = to_center.length() as f64;

            let effective_radius = match shape {
                CollisionShape::Sphere { radius } => *radius as f64,
                CollisionShape::Box { half_extents } => half_extents.length() as f64,
                CollisionShape::Capsule { radius, half_height } => {
                    (*radius + *half_height) as f64
                }
                _ => {
                    let aabb = shape.local_aabb();
                    (aabb.max - aabb.min).length() as f64 * 0.5
                }
            };

            if dist_to_center < effective_radius + margin {
                count += 1;
                velocity_sum += particle.velocity;
                position_sum += particle.position;
            }
        }
    }

    let avg_velocity = if count > 0 {
        velocity_sum / count as f32
    } else {
        Vec3::ZERO
    };

    let center_of_buoyancy = if count > 0 {
        position_sum / count as f32
    } else {
        position
    };

    (count, avg_velocity, center_of_buoyancy)
}

/// Estimate the submerged fraction of a body based on nearby particle count.
///
/// Uses the ratio of actual particles to expected particles for full submersion.
pub fn estimate_submerged_fraction(
    particle_count: usize,
    shape: &CollisionShape,
    particle_spacing: f64,
) -> f64 {
    let volume = estimate_volume(shape);
    // Expected particles for full submersion: volume / (particle_spacing^3)
    let expected_count = volume / particle_spacing.powi(3);

    if expected_count < 1.0 {
        if particle_count > 0 { 1.0 } else { 0.0 }
    } else {
        (particle_count as f64 / expected_count).min(1.0)
    }
}

/// Output of fluid force computation on a rigid body.
#[derive(Debug, Clone, Copy, Default)]
pub struct FluidForceOutput {
    /// Total force to apply at center of mass
    pub force: Vec3,
    /// Total torque to apply
    pub torque: Vec3,
    /// Buoyancy component of force
    pub buoyancy: Vec3,
    /// Drag component of force
    pub drag: Vec3,
    /// Estimated submerged fraction (0.0 - 1.0)
    pub submerged_fraction: f64,
}

/// Apply fluid forces (buoyancy + drag) to a rigid body.
///
/// # Arguments
/// * `shape` - Collision shape of the body
/// * `position` - World position of the body
/// * `rotation` - World rotation of the body
/// * `velocity` - Linear velocity of the body
/// * `particles` - Fluid particles to check for interaction
/// * `gravity` - Gravity vector
/// * `params` - Coupling parameters
///
/// # Returns
/// Combined force and torque to apply to the body
pub fn apply_fluid_forces_to_body(
    shape: &CollisionShape,
    position: Vec3,
    rotation: Quat,
    velocity: Vec3,
    particles: &[FluidParticle],
    gravity: Vec3,
    params: &FluidCouplingParams,
) -> FluidForceOutput {
    if particles.is_empty() || (!params.buoyancy_enabled && !params.drag_enabled) {
        return FluidForceOutput::default();
    }

    // Count particles near the body and get average fluid velocity
    let (count, avg_fluid_velocity, center_of_buoyancy) = count_particles_near_body(
        particles,
        shape,
        position,
        rotation,
        params.smoothing_radius,
    );

    if count == 0 {
        return FluidForceOutput::default();
    }

    // Estimate submerged fraction
    let submerged_fraction = estimate_submerged_fraction(
        count,
        shape,
        params.smoothing_radius,
    );

    let volume = estimate_volume(shape);
    let submerged_volume = volume * submerged_fraction;

    let mut total_force = Vec3::ZERO;
    let mut buoyancy = Vec3::ZERO;
    let mut drag = Vec3::ZERO;

    // Compute buoyancy
    if params.buoyancy_enabled && submerged_volume > 1e-10 {
        buoyancy = compute_buoyancy(params.fluid_density, submerged_volume, gravity);
        total_force += buoyancy;
    }

    // Compute drag
    if params.drag_enabled {
        let relative_velocity = avg_fluid_velocity - velocity;
        let cross_section = estimate_cross_section(shape, relative_velocity);
        // Scale cross-section by submerged fraction
        let effective_area = cross_section * submerged_fraction;

        drag = compute_drag(
            relative_velocity,
            params.drag_coefficient,
            params.fluid_density,
            effective_area,
        );
        total_force += drag;
    }

    // Compute torque from off-center buoyancy
    let torque = if params.buoyancy_enabled && submerged_fraction > 0.01 {
        let r = center_of_buoyancy - position;
        r.cross(buoyancy)
    } else {
        Vec3::ZERO
    };

    FluidForceOutput {
        force: total_force,
        torque,
        buoyancy,
        drag,
        submerged_fraction,
    }
}

/// Apply boundary forces from a rigid body to fluid particles.
///
/// Modifies particle accelerations in place.
///
/// # Returns
/// Total reaction force (equal and opposite to sum of boundary forces)
pub fn apply_boundary_forces_to_particles(
    particles: &mut [FluidParticle],
    shape: &CollisionShape,
    position: Vec3,
    rotation: Quat,
    body_velocity: Vec3,
    params: &FluidCouplingParams,
) -> Vec3 {
    let mut boundary = RigidBodyBoundary::with_params(
        shape.clone(),
        position,
        rotation,
        params.stiffness,
        params.damping,
    );
    boundary.velocity = body_velocity;

    let mut total_reaction = Vec3::ZERO;

    for particle in particles.iter_mut() {
        let force = boundary.apply_force(particle);
        if force.length_squared() > 0.0 {
            particle.acceleration += force / particle.mass as f32;
            total_reaction -= force; // Reaction force on body
        }
    }

    total_reaction
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    // ==================== compute_buoyancy Tests ====================

    #[test]
    fn buoyancy_zero_when_no_volume() {
        let buoyancy = compute_buoyancy(1000.0, 0.0, Vec3::new(0.0, -9.81, 0.0));
        assert!(buoyancy.length() < EPSILON);
    }

    #[test]
    fn buoyancy_zero_when_no_gravity() {
        let buoyancy = compute_buoyancy(1000.0, 1.0, Vec3::ZERO);
        assert!(buoyancy.length() < EPSILON);
    }

    #[test]
    fn buoyancy_points_up() {
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let buoyancy = compute_buoyancy(1000.0, 1.0, gravity);

        assert!(buoyancy.y > 0.0, "Buoyancy should point up");
        assert!(buoyancy.x.abs() < EPSILON);
        assert!(buoyancy.z.abs() < EPSILON);
    }

    #[test]
    fn buoyancy_magnitude_correct() {
        let gravity = Vec3::new(0.0, -10.0, 0.0);
        let buoyancy = compute_buoyancy(1000.0, 0.5, gravity);

        // Expected: 1000 * 0.5 * 10 = 5000 N
        assert!((buoyancy.y - 5000.0).abs() < 1.0);
    }

    #[test]
    fn buoyancy_proportional_to_density() {
        let gravity = Vec3::new(0.0, -10.0, 0.0);
        let b1 = compute_buoyancy(1000.0, 1.0, gravity);
        let b2 = compute_buoyancy(2000.0, 1.0, gravity);

        assert!((b2.y / b1.y - 2.0).abs() < 0.01);
    }

    // ==================== compute_drag Tests ====================

    #[test]
    fn drag_zero_when_no_velocity() {
        let drag = compute_drag(Vec3::ZERO, 0.5, 1000.0, 1.0);
        assert!(drag.length() < EPSILON);
    }

    #[test]
    fn drag_in_velocity_direction() {
        let velocity = Vec3::new(1.0, 0.0, 0.0);
        let drag = compute_drag(velocity, 0.5, 1000.0, 1.0);

        assert!(drag.x > 0.0, "Drag should be in velocity direction");
        assert!(drag.y.abs() < EPSILON);
        assert!(drag.z.abs() < EPSILON);
    }

    #[test]
    fn drag_proportional_to_velocity_squared() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(2.0, 0.0, 0.0);
        let d1 = compute_drag(v1, 0.5, 1000.0, 1.0);
        let d2 = compute_drag(v2, 0.5, 1000.0, 1.0);

        // Double velocity = 4x drag
        assert!((d2.x / d1.x - 4.0).abs() < 0.01);
    }

    #[test]
    fn drag_magnitude_correct() {
        let velocity = Vec3::new(2.0, 0.0, 0.0);
        let drag = compute_drag(velocity, 0.5, 1000.0, 1.0);

        // Expected: 0.5 * 0.5 * 1000 * 1.0 * 4 = 1000 N
        assert!((drag.x - 1000.0).abs() < 1.0);
    }

    // ==================== estimate_volume Tests ====================

    #[test]
    fn volume_sphere() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let volume = estimate_volume(&shape);
        let expected = (4.0 / 3.0) * std::f64::consts::PI;
        assert!((volume - expected).abs() < 0.01);
    }

    #[test]
    fn volume_box() {
        let shape = CollisionShape::Box { half_extents: Vec3::ONE };
        let volume = estimate_volume(&shape);
        assert!((volume - 8.0).abs() < 0.01);
    }

    #[test]
    fn volume_capsule() {
        let shape = CollisionShape::Capsule { radius: 1.0, half_height: 1.0 };
        let volume = estimate_volume(&shape);
        // Cylinder: pi * 1^2 * 2 = 2*pi
        // Sphere: (4/3) * pi * 1^3
        let expected = 2.0 * std::f64::consts::PI + (4.0 / 3.0) * std::f64::consts::PI;
        assert!((volume - expected).abs() < 0.1);
    }

    // ==================== FluidCouplingParams Tests ====================

    #[test]
    fn params_default() {
        let params = FluidCouplingParams::default();
        assert!(params.buoyancy_enabled);
        assert!(params.drag_enabled);
        assert!((params.fluid_density - 1000.0).abs() < 0.01);
    }

    #[test]
    fn params_buoyancy_only() {
        let params = FluidCouplingParams::buoyancy_only();
        assert!(params.buoyancy_enabled);
        assert!(!params.drag_enabled);
    }

    #[test]
    fn params_drag_only() {
        let params = FluidCouplingParams::drag_only();
        assert!(!params.buoyancy_enabled);
        assert!(params.drag_enabled);
    }

    #[test]
    fn params_disabled() {
        let params = FluidCouplingParams::disabled();
        assert!(!params.buoyancy_enabled);
        assert!(!params.drag_enabled);
    }

    // ==================== apply_fluid_forces_to_body Tests ====================

    #[test]
    fn fluid_forces_zero_when_no_particles() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let params = FluidCouplingParams::default();
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        let output = apply_fluid_forces_to_body(
            &shape,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ZERO,
            &[],
            gravity,
            &params,
        );

        assert!(output.force.length() < EPSILON);
        assert!(output.torque.length() < EPSILON);
    }

    #[test]
    fn fluid_forces_zero_when_disabled() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let params = FluidCouplingParams::disabled();
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        // Particles inside the sphere
        let particles = vec![
            FluidParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            FluidParticle::new(Vec3::new(0.1, 0.0, 0.0), 1.0),
        ];

        let output = apply_fluid_forces_to_body(
            &shape,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ZERO,
            &particles,
            gravity,
            &params,
        );

        assert!(output.force.length() < EPSILON);
    }

    #[test]
    fn fluid_forces_buoyancy_when_submerged() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let mut params = FluidCouplingParams::buoyancy_only();
        params.smoothing_radius = 2.0; // Large radius to catch all particles
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        // Create particles inside and around the sphere
        let mut particles = Vec::new();
        for x in -2..=2 {
            for y in -2..=2 {
                for z in -2..=2 {
                    particles.push(FluidParticle::new(
                        Vec3::new(x as f32 * 0.3, y as f32 * 0.3, z as f32 * 0.3),
                        1.0,
                    ));
                }
            }
        }

        let output = apply_fluid_forces_to_body(
            &shape,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ZERO,
            &particles,
            gravity,
            &params,
        );

        assert!(output.buoyancy.y > 0.0, "Should have upward buoyancy");
        assert!(output.submerged_fraction > 0.0, "Should be partially submerged");
    }

    #[test]
    fn fluid_forces_drag_when_moving() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let mut params = FluidCouplingParams::drag_only();
        params.smoothing_radius = 2.0;
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        // Stationary particles
        let mut particles = Vec::new();
        for x in -2..=2 {
            for y in -2..=2 {
                for z in -2..=2 {
                    particles.push(FluidParticle::new(
                        Vec3::new(x as f32 * 0.3, y as f32 * 0.3, z as f32 * 0.3),
                        1.0,
                    ));
                }
            }
        }

        // Body moving through fluid
        let body_velocity = Vec3::new(5.0, 0.0, 0.0);

        let output = apply_fluid_forces_to_body(
            &shape,
            Vec3::ZERO,
            Quat::IDENTITY,
            body_velocity,
            &particles,
            gravity,
            &params,
        );

        // Drag should oppose body motion
        assert!(output.drag.x < 0.0, "Drag should oppose body motion");
    }

    // ==================== apply_boundary_forces_to_particles Tests ====================

    #[test]
    fn boundary_forces_push_particles_out() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let params = FluidCouplingParams::default();

        // Particle inside the sphere
        let mut particles = vec![FluidParticle::new(Vec3::new(0.5, 0.0, 0.0), 1.0)];
        particles[0].acceleration = Vec3::ZERO;

        let _reaction = apply_boundary_forces_to_particles(
            &mut particles,
            &shape,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ZERO,
            &params,
        );

        // Particle should be pushed outward
        assert!(particles[0].acceleration.x > 0.0, "Should push particle out (+X)");
    }

    #[test]
    fn boundary_forces_return_reaction() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let params = FluidCouplingParams::default();

        let mut particles = vec![FluidParticle::new(Vec3::new(0.5, 0.0, 0.0), 1.0)];
        particles[0].acceleration = Vec3::ZERO;

        let reaction = apply_boundary_forces_to_particles(
            &mut particles,
            &shape,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ZERO,
            &params,
        );

        // Reaction should be opposite to particle acceleration
        assert!(reaction.x < 0.0, "Reaction should be opposite to boundary force");
    }

    #[test]
    fn boundary_forces_zero_when_outside() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let params = FluidCouplingParams::default();

        // Particle outside the sphere
        let mut particles = vec![FluidParticle::new(Vec3::new(2.0, 0.0, 0.0), 1.0)];
        particles[0].acceleration = Vec3::ZERO;

        let reaction = apply_boundary_forces_to_particles(
            &mut particles,
            &shape,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ZERO,
            &params,
        );

        assert!(particles[0].acceleration.length() < EPSILON);
        assert!(reaction.length() < EPSILON);
    }
}
