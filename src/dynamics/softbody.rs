//! Soft body dynamics using mass-spring systems.
//!
//! Supports cloth, jelly, and other deformable objects.

use glam::Vec3;

/// A particle in a soft body (mass point).
#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub force: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
    pub pinned: bool,
}

impl Particle {
    /// Create a new particle.
    pub fn new(position: Vec3, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            velocity: Vec3::ZERO,
            force: Vec3::ZERO,
            mass,
            inv_mass,
            pinned: false,
        }
    }

    /// Create a pinned particle (won't move).
    pub fn pinned(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            force: Vec3::ZERO,
            mass: 0.0,
            inv_mass: 0.0,
            pinned: true,
        }
    }

    /// Apply a force to this particle.
    pub fn apply_force(&mut self, force: Vec3) {
        if !self.pinned {
            self.force += force;
        }
    }

    /// Clear accumulated forces.
    pub fn clear_forces(&mut self) {
        self.force = Vec3::ZERO;
    }
}

/// A spring connecting two particles.
#[derive(Debug, Clone, Copy)]
pub struct Spring {
    /// Index of first particle.
    pub p1: usize,
    /// Index of second particle.
    pub p2: usize,
    /// Rest length of the spring.
    pub rest_length: f32,
    /// Spring stiffness coefficient.
    pub stiffness: f32,
    /// Damping coefficient.
    pub damping: f32,
}

impl Spring {
    /// Create a new spring.
    pub fn new(p1: usize, p2: usize, rest_length: f32, stiffness: f32, damping: f32) -> Self {
        Self {
            p1,
            p2,
            rest_length,
            stiffness,
            damping,
        }
    }

    /// Calculate spring force and apply to particles.
    pub fn apply(&self, particles: &mut [Particle]) {
        let pos1 = particles[self.p1].position;
        let pos2 = particles[self.p2].position;
        let vel1 = particles[self.p1].velocity;
        let vel2 = particles[self.p2].velocity;

        let delta = pos2 - pos1;
        let length = delta.length();

        if length < 1e-6 {
            return; // Avoid division by zero
        }

        let direction = delta / length;

        // Hooke's law: F = -k * (length - rest_length)
        let stretch = length - self.rest_length;
        let spring_force = self.stiffness * stretch;

        // Damping: F = -c * relative_velocity_along_spring
        let relative_vel = vel2 - vel1;
        let damping_force = self.damping * relative_vel.dot(direction);

        let total_force = (spring_force + damping_force) * direction;

        particles[self.p1].apply_force(total_force);
        particles[self.p2].apply_force(-total_force);
    }
}

/// Soft body configuration.
#[derive(Debug, Clone)]
pub struct SoftBodyConfig {
    /// Spring stiffness for structural springs.
    pub structural_stiffness: f32,
    /// Spring stiffness for shear springs.
    pub shear_stiffness: f32,
    /// Spring stiffness for bend springs.
    pub bend_stiffness: f32,
    /// Damping coefficient.
    pub damping: f32,
    /// Pressure coefficient for volume preservation.
    pub pressure: f32,
}

impl Default for SoftBodyConfig {
    fn default() -> Self {
        Self {
            structural_stiffness: 100.0,
            shear_stiffness: 50.0,
            bend_stiffness: 10.0,
            damping: 0.5,
            pressure: 0.0,
        }
    }
}

impl SoftBodyConfig {
    /// Config for cloth-like behavior.
    pub fn cloth() -> Self {
        Self {
            structural_stiffness: 200.0,
            shear_stiffness: 50.0,
            bend_stiffness: 5.0,
            damping: 0.3,
            pressure: 0.0,
        }
    }

    /// Config for jelly-like behavior.
    pub fn jelly() -> Self {
        Self {
            structural_stiffness: 50.0,
            shear_stiffness: 30.0,
            bend_stiffness: 20.0,
            damping: 0.8,
            pressure: 10.0,
        }
    }
}

/// A soft body made of particles and springs.
#[derive(Debug, Clone)]
pub struct SoftBody {
    pub particles: Vec<Particle>,
    pub springs: Vec<Spring>,
    pub config: SoftBodyConfig,
    rest_volume: f32,
}

impl SoftBody {
    /// Create an empty soft body.
    pub fn new(config: SoftBodyConfig) -> Self {
        Self {
            particles: Vec::new(),
            springs: Vec::new(),
            config,
            rest_volume: 0.0,
        }
    }

    /// Add a particle and return its index.
    pub fn add_particle(&mut self, position: Vec3, mass: f32) -> usize {
        let idx = self.particles.len();
        self.particles.push(Particle::new(position, mass));
        idx
    }

    /// Add a pinned particle and return its index.
    pub fn add_pinned(&mut self, position: Vec3) -> usize {
        let idx = self.particles.len();
        self.particles.push(Particle::pinned(position));
        idx
    }

    /// Add a spring between two particles.
    pub fn add_spring(&mut self, p1: usize, p2: usize, stiffness: f32, damping: f32) {
        let rest_length = (self.particles[p1].position - self.particles[p2].position).length();
        self.springs.push(Spring::new(p1, p2, rest_length, stiffness, damping));
    }

    /// Add a structural spring (uses config stiffness).
    pub fn add_structural(&mut self, p1: usize, p2: usize) {
        self.add_spring(p1, p2, self.config.structural_stiffness, self.config.damping);
    }

    /// Add a shear spring (uses config stiffness).
    pub fn add_shear(&mut self, p1: usize, p2: usize) {
        self.add_spring(p1, p2, self.config.shear_stiffness, self.config.damping);
    }

    /// Add a bend spring (uses config stiffness).
    pub fn add_bend(&mut self, p1: usize, p2: usize) {
        self.add_spring(p1, p2, self.config.bend_stiffness, self.config.damping);
    }

    /// Calculate and store rest volume for pressure.
    pub fn compute_rest_volume(&mut self) {
        self.rest_volume = self.calculate_volume();
    }

    /// Calculate current volume (approximation using centroid).
    pub fn calculate_volume(&self) -> f32 {
        if self.particles.len() < 4 {
            return 0.0;
        }

        // Approximate as convex hull volume using centroid
        let centroid = self.centroid();
        let mut volume = 0.0;

        // Sum tetrahedron volumes from centroid to each triangle
        // This is a rough approximation
        for i in 0..self.particles.len() {
            let j = (i + 1) % self.particles.len();
            let k = (i + 2) % self.particles.len();

            let a = self.particles[i].position - centroid;
            let b = self.particles[j].position - centroid;
            let c = self.particles[k].position - centroid;

            volume += a.cross(b).dot(c).abs() / 6.0;
        }

        volume
    }

    /// Get centroid of all particles.
    pub fn centroid(&self) -> Vec3 {
        if self.particles.is_empty() {
            return Vec3::ZERO;
        }

        let sum: Vec3 = self.particles.iter().map(|p| p.position).sum();
        sum / self.particles.len() as f32
    }

    /// Apply spring forces.
    pub fn apply_spring_forces(&mut self) {
        for spring in &self.springs {
            spring.apply(&mut self.particles);
        }
    }

    /// Apply pressure force for volume preservation.
    pub fn apply_pressure(&mut self) {
        if self.config.pressure <= 0.0 || self.rest_volume <= 0.0 {
            return;
        }

        let current_volume = self.calculate_volume();
        let volume_diff = self.rest_volume - current_volume;
        let pressure = self.config.pressure * volume_diff / self.rest_volume;

        let centroid = self.centroid();

        for particle in &mut self.particles {
            if !particle.pinned {
                let direction = (particle.position - centroid).normalize_or_zero();
                particle.apply_force(direction * pressure);
            }
        }
    }

    /// Apply gravity to all particles.
    pub fn apply_gravity(&mut self, gravity: Vec3) {
        for particle in &mut self.particles {
            particle.apply_force(gravity * particle.mass);
        }
    }

    /// Integrate particle positions using Verlet integration.
    pub fn integrate(&mut self, dt: f32) {
        for particle in &mut self.particles {
            if particle.pinned {
                continue;
            }

            // Semi-implicit Euler integration
            let acceleration = particle.force * particle.inv_mass;
            particle.velocity += acceleration * dt;
            particle.position += particle.velocity * dt;
            particle.clear_forces();
        }
    }

    /// Step the soft body simulation.
    pub fn step(&mut self, dt: f32, gravity: Vec3) {
        self.apply_gravity(gravity);
        self.apply_spring_forces();
        self.apply_pressure();
        self.integrate(dt);
    }

    /// Check collision with a plane and respond.
    pub fn collide_plane(&mut self, normal: Vec3, offset: f32, restitution: f32) {
        for particle in &mut self.particles {
            if particle.pinned {
                continue;
            }

            let dist = particle.position.dot(normal) - offset;
            if dist < 0.0 {
                // Push out of plane
                particle.position -= normal * dist;

                // Reflect velocity
                let vn = particle.velocity.dot(normal);
                if vn < 0.0 {
                    particle.velocity -= normal * (1.0 + restitution) * vn;
                }
            }
        }
    }

    /// Check collision with a sphere and respond.
    pub fn collide_sphere(&mut self, center: Vec3, radius: f32, restitution: f32) {
        for particle in &mut self.particles {
            if particle.pinned {
                continue;
            }

            let delta = particle.position - center;
            let dist = delta.length();

            if dist < radius && dist > 1e-6 {
                let normal = delta / dist;
                let penetration = radius - dist;

                // Push out of sphere
                particle.position += normal * penetration;

                // Reflect velocity
                let vn = particle.velocity.dot(normal);
                if vn < 0.0 {
                    particle.velocity -= normal * (1.0 + restitution) * vn;
                }
            }
        }
    }

    /// Create a cloth grid.
    pub fn create_cloth(width: usize, height: usize, spacing: f32, config: SoftBodyConfig) -> Self {
        let mut body = Self::new(config);

        // Create particles in a grid
        for y in 0..height {
            for x in 0..width {
                let pos = Vec3::new(x as f32 * spacing, 0.0, y as f32 * spacing);
                body.add_particle(pos, 1.0);
            }
        }

        // Add structural springs (horizontal and vertical)
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;

                // Right neighbor
                if x + 1 < width {
                    body.add_structural(idx, idx + 1);
                }

                // Bottom neighbor
                if y + 1 < height {
                    body.add_structural(idx, idx + width);
                }
            }
        }

        // Add shear springs (diagonals)
        for y in 0..height - 1 {
            for x in 0..width - 1 {
                let idx = y * width + x;
                body.add_shear(idx, idx + width + 1);
                body.add_shear(idx + 1, idx + width);
            }
        }

        // Add bend springs (skip one particle)
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;

                if x + 2 < width {
                    body.add_bend(idx, idx + 2);
                }
                if y + 2 < height {
                    body.add_bend(idx, idx + 2 * width);
                }
            }
        }

        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_new() {
        let p = Particle::new(Vec3::new(1.0, 2.0, 3.0), 2.0);
        assert_eq!(p.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(p.mass, 2.0);
        assert!((p.inv_mass - 0.5).abs() < 1e-6);
        assert!(!p.pinned);
    }

    #[test]
    fn test_particle_pinned() {
        let p = Particle::pinned(Vec3::ZERO);
        assert!(p.pinned);
        assert_eq!(p.inv_mass, 0.0);
    }

    #[test]
    fn test_particle_apply_force() {
        let mut p = Particle::new(Vec3::ZERO, 1.0);
        p.apply_force(Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(p.force, Vec3::new(10.0, 0.0, 0.0));

        // Pinned particle ignores force
        let mut pinned = Particle::pinned(Vec3::ZERO);
        pinned.apply_force(Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(pinned.force, Vec3::ZERO);
    }

    #[test]
    fn test_spring_force() {
        let mut particles = vec![
            Particle::new(Vec3::ZERO, 1.0),
            Particle::new(Vec3::new(2.0, 0.0, 0.0), 1.0), // Stretched beyond rest length
        ];

        let spring = Spring::new(0, 1, 1.0, 10.0, 0.0); // Rest length 1, current length 2
        spring.apply(&mut particles);

        // p0 should be pulled toward p1 (positive x)
        assert!(particles[0].force.x > 0.0);
        // p1 should be pulled toward p0 (negative x)
        assert!(particles[1].force.x < 0.0);
    }

    #[test]
    fn test_soft_body_add_particles() {
        let mut body = SoftBody::new(SoftBodyConfig::default());
        let idx1 = body.add_particle(Vec3::ZERO, 1.0);
        let idx2 = body.add_particle(Vec3::X, 1.0);

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(body.particles.len(), 2);
    }

    #[test]
    fn test_soft_body_centroid() {
        let mut body = SoftBody::new(SoftBodyConfig::default());
        body.add_particle(Vec3::ZERO, 1.0);
        body.add_particle(Vec3::new(2.0, 0.0, 0.0), 1.0);

        let centroid = body.centroid();
        assert!((centroid.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_soft_body_integrate() {
        let mut body = SoftBody::new(SoftBodyConfig::default());
        body.add_particle(Vec3::ZERO, 1.0);
        body.particles[0].apply_force(Vec3::new(10.0, 0.0, 0.0));

        body.integrate(0.1);

        // Position should have changed
        assert!(body.particles[0].position.x > 0.0);
    }

    #[test]
    fn test_create_cloth() {
        let cloth = SoftBody::create_cloth(3, 3, 1.0, SoftBodyConfig::cloth());

        assert_eq!(cloth.particles.len(), 9); // 3x3 grid

        // Should have structural, shear, and bend springs
        assert!(!cloth.springs.is_empty());
    }

    #[test]
    fn test_collide_plane() {
        let mut body = SoftBody::new(SoftBodyConfig::default());
        body.add_particle(Vec3::new(0.0, -1.0, 0.0), 1.0); // Below ground
        body.particles[0].velocity = Vec3::new(0.0, -5.0, 0.0);

        body.collide_plane(Vec3::Y, 0.0, 0.5); // Ground at y=0

        // Should be pushed up to y=0
        assert!(body.particles[0].position.y >= 0.0);
        // Velocity should be reflected
        assert!(body.particles[0].velocity.y > 0.0);
    }

    #[test]
    fn test_config_presets() {
        let cloth = SoftBodyConfig::cloth();
        let jelly = SoftBodyConfig::jelly();

        // Cloth is stiffer structurally
        assert!(cloth.structural_stiffness > jelly.structural_stiffness);
        // Jelly has pressure
        assert!(jelly.pressure > 0.0);
    }

    #[test]
    fn test_soft_body_step() {
        let mut body = SoftBody::new(SoftBodyConfig::default());
        body.add_particle(Vec3::new(0.0, 10.0, 0.0), 1.0);

        let initial_y = body.particles[0].position.y;
        body.step(0.1, Vec3::new(0.0, -9.8, 0.0));

        // Should have fallen
        assert!(body.particles[0].position.y < initial_y);
    }
}
