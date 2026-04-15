//! SPH fluid solver implementation.
//!
//! This module provides the main SPH solver that orchestrates the fluid simulation:
//! - Density computation using Poly6 kernel
//! - Pressure computation via Tait equation
//! - Force computation (pressure + viscosity + gravity)
//! - Symplectic Euler integration

use glam::Vec3;

use super::kernel::Kernels;
use super::particle::{FluidParticle, ParticleGrid};

/// SPH fluid solver.
///
/// Manages a collection of fluid particles and simulates their behavior using
/// Smoothed Particle Hydrodynamics with standard SPH kernels.
#[derive(Debug, Clone)]
pub struct SPHSolver {
    /// Fluid particles
    particles: Vec<FluidParticle>,
    /// Precomputed kernel constants
    kernels: Kernels,
    /// Spatial hash grid for neighbor queries
    grid: ParticleGrid,
    /// Gravity acceleration vector
    pub gravity: Vec3,
    /// Rest density of the fluid (kg/m^3)
    pub rest_density: f64,
    /// Gas constant for Tait equation (stiffness)
    pub gas_constant: f64,
    /// Viscosity coefficient
    pub viscosity_coeff: f64,
}

impl SPHSolver {
    /// Create a new SPH solver with the given smoothing radius.
    ///
    /// Uses default values:
    /// - gravity: (0, -9.81, 0)
    /// - rest_density: 1000.0 kg/m^3 (water)
    /// - gas_constant: 2000.0 (Tait equation stiffness)
    /// - viscosity_coeff: 200.0
    pub fn new(h: f64) -> Self {
        Self {
            particles: Vec::new(),
            kernels: Kernels::new(h),
            grid: ParticleGrid::new(h),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            rest_density: 1000.0,
            gas_constant: 2000.0,
            viscosity_coeff: 200.0,
        }
    }

    /// Add a particle to the simulation.
    pub fn add_particle(&mut self, particle: FluidParticle) {
        self.particles.push(particle);
    }

    /// Get the number of particles in the simulation.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Get a reference to the particles.
    pub fn particles(&self) -> &[FluidParticle] {
        &self.particles
    }

    /// Get a mutable reference to the particles.
    pub fn particles_mut(&mut self) -> &mut [FluidParticle] {
        &mut self.particles
    }

    /// Get the smoothing radius.
    pub fn smoothing_radius(&self) -> f64 {
        self.kernels.h
    }

    /// Perform one simulation step with the given time delta.
    ///
    /// The simulation step consists of:
    /// 1. Build spatial hash grid from current positions
    /// 2. Compute density for each particle
    /// 3. Compute pressure via Tait equation
    /// 4. Compute forces (pressure + viscosity + gravity)
    /// 5. Integrate with symplectic Euler
    pub fn step(&mut self, dt: f64) {
        if self.particles.is_empty() {
            return;
        }

        self.build_grid();
        self.compute_density();
        self.compute_pressure();
        self.compute_forces();
        self.integrate(dt);
    }

    /// Build the spatial hash grid from current particle positions.
    fn build_grid(&mut self) {
        self.grid.clear();
        for (i, particle) in self.particles.iter().enumerate() {
            self.grid.insert(i, particle.position);
        }
    }

    /// Compute density for each particle using Poly6 kernel.
    fn compute_density(&mut self) {
        let h = self.kernels.h;
        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.position).collect();
        let masses: Vec<f64> = self.particles.iter().map(|p| p.mass).collect();

        for i in 0..self.particles.len() {
            let pos_i = positions[i];
            let neighbors = self.grid.query(pos_i, h);

            let mut density = 0.0;
            for &j in &neighbors {
                let r_vec = pos_i - positions[j];
                let r = r_vec.length() as f64;
                density += masses[j] * self.kernels.poly6(r);
            }

            self.particles[i].density = density;
        }
    }

    /// Compute pressure using Tait equation: p = k * (rho - rho_0)
    fn compute_pressure(&mut self) {
        for particle in &mut self.particles {
            particle.pressure = self.gas_constant * (particle.density - self.rest_density);
        }
    }

    /// Compute forces: pressure + viscosity + gravity.
    fn compute_forces(&mut self) {
        let h = self.kernels.h;
        let positions: Vec<Vec3> = self.particles.iter().map(|p| p.position).collect();
        let velocities: Vec<Vec3> = self.particles.iter().map(|p| p.velocity).collect();
        let densities: Vec<f64> = self.particles.iter().map(|p| p.density).collect();
        let pressures: Vec<f64> = self.particles.iter().map(|p| p.pressure).collect();
        let masses: Vec<f64> = self.particles.iter().map(|p| p.mass).collect();

        for i in 0..self.particles.len() {
            let pos_i = positions[i];
            let vel_i = velocities[i];
            let density_i = densities[i];
            let pressure_i = pressures[i];

            // Skip if density is too low (avoid division by zero)
            if density_i < 1e-10 {
                self.particles[i].acceleration = self.gravity;
                continue;
            }

            let neighbors = self.grid.query(pos_i, h);

            let mut pressure_force = Vec3::ZERO;
            let mut viscosity_force = Vec3::ZERO;

            for &j in &neighbors {
                if i == j {
                    continue;
                }

                let pos_j = positions[j];
                let r_vec = pos_i - pos_j;
                let r = r_vec.length() as f64;

                // Skip if particles are at the same position
                if r < 1e-10 {
                    continue;
                }

                let density_j = densities[j];
                if density_j < 1e-10 {
                    continue;
                }

                // Pressure force: -sum_j m_j * (p_i + p_j) / (2 * rho_j) * grad W
                let pressure_term = masses[j] * (pressure_i + pressures[j]) / (2.0 * density_j);
                let grad = self.kernels.spiky_grad(r_vec);
                pressure_force -= grad * pressure_term as f32;

                // Viscosity force: mu * sum_j m_j * (v_j - v_i) / rho_j * laplacian W
                let vel_j = velocities[j];
                let laplacian = self.kernels.visc_laplacian(r);
                let visc_term = masses[j] * laplacian / density_j;
                viscosity_force += (vel_j - vel_i) * visc_term as f32;
            }

            viscosity_force *= self.viscosity_coeff as f32;

            // Total acceleration: (F_pressure + F_viscosity) / rho + gravity
            let acceleration = (pressure_force + viscosity_force) / density_i as f32 + self.gravity;
            self.particles[i].acceleration = acceleration;
        }
    }

    /// Integrate using symplectic Euler: v += a*dt, x += v*dt
    fn integrate(&mut self, dt: f64) {
        let dt_f32 = dt as f32;
        for particle in &mut self.particles {
            particle.velocity += particle.acceleration * dt_f32;
            particle.position += particle.velocity * dt_f32;
        }
    }
}

/// Convenience wrapper around SPHSolver for simpler fluid simulation.
///
/// FluidWorld provides a streamlined interface for common use cases.
#[derive(Debug, Clone)]
pub struct FluidWorld {
    solver: SPHSolver,
}

impl FluidWorld {
    /// Create a new fluid world with the given smoothing radius.
    pub fn new(h: f64) -> Self {
        Self {
            solver: SPHSolver::new(h),
        }
    }

    /// Add a particle to the simulation.
    pub fn add_particle(&mut self, particle: FluidParticle) {
        self.solver.add_particle(particle);
    }

    /// Perform one simulation step.
    pub fn step(&mut self, dt: f64) {
        self.solver.step(dt);
    }

    /// Get a reference to the particles.
    pub fn particles(&self) -> &[FluidParticle] {
        self.solver.particles()
    }

    /// Get a mutable reference to the underlying solver.
    pub fn solver_mut(&mut self) -> &mut SPHSolver {
        &mut self.solver
    }

    /// Get a reference to the underlying solver.
    pub fn solver(&self) -> &SPHSolver {
        &self.solver
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;
    const H: f64 = 1.0;

    // ==================== SPHSolver Construction Tests ====================

    #[test]
    fn solver_new_has_default_values() {
        let solver = SPHSolver::new(H);

        assert_eq!(solver.particle_count(), 0);
        assert!((solver.gravity - Vec3::new(0.0, -9.81, 0.0)).length() < EPSILON);
        assert!((solver.rest_density - 1000.0).abs() < 1e-10);
        assert!((solver.gas_constant - 2000.0).abs() < 1e-10);
        assert!((solver.viscosity_coeff - 200.0).abs() < 1e-10);
        assert!((solver.smoothing_radius() - H).abs() < 1e-10);
    }

    #[test]
    fn solver_add_particle_increments_count() {
        let mut solver = SPHSolver::new(H);
        assert_eq!(solver.particle_count(), 0);

        solver.add_particle(FluidParticle::new(Vec3::ZERO, 1.0));
        assert_eq!(solver.particle_count(), 1);

        solver.add_particle(FluidParticle::new(Vec3::ONE, 1.0));
        assert_eq!(solver.particle_count(), 2);
    }

    #[test]
    fn solver_particles_accessor() {
        let mut solver = SPHSolver::new(H);
        let p1 = FluidParticle::new(Vec3::new(1.0, 2.0, 3.0), 0.5);
        let p2 = FluidParticle::new(Vec3::new(4.0, 5.0, 6.0), 1.5);

        solver.add_particle(p1);
        solver.add_particle(p2);

        let particles = solver.particles();
        assert_eq!(particles.len(), 2);
        assert!((particles[0].position - Vec3::new(1.0, 2.0, 3.0)).length() < EPSILON);
        assert!((particles[1].position - Vec3::new(4.0, 5.0, 6.0)).length() < EPSILON);
    }

    // ==================== Step Function Tests ====================

    #[test]
    fn solver_step_empty_does_not_panic() {
        let mut solver = SPHSolver::new(H);
        solver.step(0.01);
        assert_eq!(solver.particle_count(), 0);
    }

    #[test]
    fn solver_step_single_particle_falls_due_to_gravity() {
        let mut solver = SPHSolver::new(H);
        solver.add_particle(FluidParticle::new(Vec3::new(0.0, 10.0, 0.0), 1.0));

        let initial_y = solver.particles()[0].position.y;

        // Run several steps
        for _ in 0..10 {
            solver.step(0.01);
        }

        let final_y = solver.particles()[0].position.y;
        assert!(final_y < initial_y, "Particle should fall due to gravity");
    }

    #[test]
    fn solver_step_updates_velocity() {
        let mut solver = SPHSolver::new(H);
        solver.add_particle(FluidParticle::new(Vec3::ZERO, 1.0));

        assert!((solver.particles()[0].velocity).length() < EPSILON);

        solver.step(0.01);

        // After one step, velocity should be non-zero due to gravity
        assert!(solver.particles()[0].velocity.y < 0.0);
    }

    #[test]
    fn solver_step_computes_density_for_nearby_particles() {
        let mut solver = SPHSolver::new(H);

        // Place two particles close together
        solver.add_particle(FluidParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0));
        solver.add_particle(FluidParticle::new(Vec3::new(0.3, 0.0, 0.0), 1.0));

        solver.step(0.001);

        // Both particles should have positive density
        assert!(solver.particles()[0].density > 0.0);
        assert!(solver.particles()[1].density > 0.0);
    }

    #[test]
    fn solver_step_computes_pressure() {
        let mut solver = SPHSolver::new(H);
        solver.rest_density = 100.0; // Lower rest density for testing

        // Place particles in a dense cluster
        for i in 0..5 {
            solver.add_particle(FluidParticle::new(
                Vec3::new(i as f32 * 0.2, 0.0, 0.0),
                1.0,
            ));
        }

        solver.step(0.001);

        // Particles in the middle should have pressure computed
        // Pressure = gas_constant * (density - rest_density)
        for particle in solver.particles() {
            // Pressure can be positive or negative depending on density vs rest_density
            assert!(particle.pressure.is_finite());
        }
    }

    // ==================== Integration Tests ====================

    #[test]
    fn solver_symplectic_euler_correct() {
        let mut solver = SPHSolver::new(H);
        solver.gravity = Vec3::new(0.0, -10.0, 0.0); // Simplified gravity

        let p = FluidParticle::with_velocity(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 1.0);
        solver.add_particle(p);

        let dt = 0.1;
        solver.step(dt);

        // For isolated particle: a = gravity, v_new = v + a*dt, x_new = x + v_new*dt
        // v_new = (1, 0, 0) + (0, -10, 0) * 0.1 = (1, -1, 0)
        // x_new = (0, 0, 0) + (1, -1, 0) * 0.1 = (0.1, -0.1, 0)
        let particle = &solver.particles()[0];
        assert!(
            (particle.velocity - Vec3::new(1.0, -1.0, 0.0)).length() < 0.01,
            "velocity: {:?}",
            particle.velocity
        );
        assert!(
            (particle.position - Vec3::new(0.1, -0.1, 0.0)).length() < 0.01,
            "position: {:?}",
            particle.position
        );
    }

    #[test]
    fn solver_particles_repel_under_high_pressure() {
        let mut solver = SPHSolver::new(0.5);
        solver.gravity = Vec3::ZERO; // Disable gravity
        solver.rest_density = 1.0;
        solver.gas_constant = 100.0;

        // Place two particles very close together
        solver.add_particle(FluidParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0));
        solver.add_particle(FluidParticle::new(Vec3::new(0.1, 0.0, 0.0), 1.0));

        let initial_dist = (solver.particles()[0].position - solver.particles()[1].position).length();

        // Run simulation
        for _ in 0..50 {
            solver.step(0.001);
        }

        let final_dist = (solver.particles()[0].position - solver.particles()[1].position).length();

        // Particles should repel each other due to pressure
        assert!(
            final_dist > initial_dist,
            "Particles should repel: initial {} -> final {}",
            initial_dist,
            final_dist
        );
    }

    // ==================== FluidWorld Tests ====================

    #[test]
    fn fluid_world_wraps_solver() {
        let mut world = FluidWorld::new(H);

        world.add_particle(FluidParticle::new(Vec3::new(1.0, 2.0, 3.0), 1.0));
        assert_eq!(world.particles().len(), 1);

        world.step(0.01);
        // Particle should have moved
        assert!((world.particles()[0].position - Vec3::new(1.0, 2.0, 3.0)).length() > EPSILON);
    }

    #[test]
    fn fluid_world_solver_access() {
        let mut world = FluidWorld::new(H);

        // Modify solver through mutable access
        world.solver_mut().gravity = Vec3::new(0.0, -5.0, 0.0);

        assert!((world.solver().gravity - Vec3::new(0.0, -5.0, 0.0)).length() < EPSILON);
    }

    #[test]
    fn fluid_world_multiple_particles_interact() {
        let mut world = FluidWorld::new(0.5);

        // Create a small grid of particles
        for x in 0..3 {
            for y in 0..3 {
                world.add_particle(FluidParticle::new(
                    Vec3::new(x as f32 * 0.2, y as f32 * 0.2, 0.0),
                    1.0,
                ));
            }
        }

        assert_eq!(world.particles().len(), 9);

        // Step the simulation
        for _ in 0..10 {
            world.step(0.001);
        }

        // All particles should still exist and have valid positions
        for particle in world.particles() {
            assert!(particle.position.x.is_finite());
            assert!(particle.position.y.is_finite());
            assert!(particle.position.z.is_finite());
        }
    }

    // ==================== Density Computation Tests ====================

    #[test]
    fn density_increases_with_particle_count() {
        // More particles nearby = higher density
        let mut solver1 = SPHSolver::new(H);
        solver1.add_particle(FluidParticle::new(Vec3::ZERO, 1.0));
        solver1.step(0.001);
        let density1 = solver1.particles()[0].density;

        let mut solver2 = SPHSolver::new(H);
        solver2.add_particle(FluidParticle::new(Vec3::ZERO, 1.0));
        solver2.add_particle(FluidParticle::new(Vec3::new(0.3, 0.0, 0.0), 1.0));
        solver2.add_particle(FluidParticle::new(Vec3::new(0.0, 0.3, 0.0), 1.0));
        solver2.step(0.001);
        let density2 = solver2.particles()[0].density;

        assert!(
            density2 > density1,
            "More neighbors should increase density: {} vs {}",
            density1,
            density2
        );
    }

    #[test]
    fn viscosity_smooths_velocity_differences() {
        let mut solver = SPHSolver::new(0.5);
        solver.gravity = Vec3::ZERO;
        solver.gas_constant = 0.0; // Disable pressure to isolate viscosity effect
        solver.viscosity_coeff = 50.0; // Moderate viscosity

        // Two particles with different velocities, close enough to interact
        solver.add_particle(FluidParticle::with_velocity(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
        ));
        solver.add_particle(FluidParticle::with_velocity(
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            1.0,
        ));

        let initial_diff = (solver.particles()[0].velocity.x - solver.particles()[1].velocity.x).abs();

        // Run simulation with very small time steps
        for _ in 0..10 {
            solver.step(0.00001);
        }

        let final_diff = (solver.particles()[0].velocity.x - solver.particles()[1].velocity.x).abs();

        // Velocity difference should decrease due to viscosity
        assert!(
            final_diff < initial_diff,
            "Viscosity should smooth velocities: {} -> {}",
            initial_diff,
            final_diff
        );
    }
}
