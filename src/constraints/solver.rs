// Sequential impulse constraint solver

use glam::{Quat, Vec3};
use super::contact::ContactConstraint;
use super::joint::Joint;
use crate::world::BodyHandle;

/// Position correction method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionCorrection {
    /// Baumgarte stabilization - adds bias to velocity solver.
    /// Simple but can add energy to the system.
    Baumgarte,
    /// Split impulse - separates position correction from velocity.
    /// More stable, prevents energy gain, but slightly more expensive.
    SplitImpulse,
    /// Non-linear Gauss-Seidel position correction.
    /// Most accurate but most expensive.
    NGS,
}

/// Solver configuration
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Number of velocity iterations
    pub velocity_iterations: usize,
    /// Number of position iterations
    pub position_iterations: usize,
    /// Baumgarte stabilization factor (used when position_correction = Baumgarte)
    pub baumgarte: f32,
    /// Slop for penetration allowance
    pub slop: f32,
    /// Enable warm starting
    pub warm_starting: bool,
    /// Position correction method
    pub position_correction: PositionCorrection,
    /// Split impulse bias factor (used when position_correction = SplitImpulse)
    pub split_impulse_beta: f32,
    /// Split impulse penetration threshold (only correct if deeper than this)
    pub split_impulse_threshold: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            velocity_iterations: 8,
            position_iterations: 3,
            baumgarte: 0.2,
            slop: 0.005,
            warm_starting: true,
            position_correction: PositionCorrection::SplitImpulse,
            split_impulse_beta: 0.2,
            split_impulse_threshold: -0.04, // 4cm
        }
    }
}

/// Body state for solver
#[derive(Debug, Clone, Copy)]
pub struct SolverBody {
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub inv_mass: f32,
    pub inv_inertia: Vec3,
    /// Split impulse pseudo-velocity (for position correction only)
    pub pseudo_velocity: Vec3,
    /// Split impulse pseudo-angular velocity
    pub pseudo_angular_velocity: Vec3,
}

impl SolverBody {
    pub fn new(
        position: Vec3,
        rotation: Quat,
        velocity: Vec3,
        angular_velocity: Vec3,
        inv_mass: f32,
        inv_inertia: Vec3,
    ) -> Self {
        Self {
            position,
            rotation,
            velocity,
            angular_velocity,
            inv_mass,
            inv_inertia,
            pseudo_velocity: Vec3::ZERO,
            pseudo_angular_velocity: Vec3::ZERO,
        }
    }
    
    pub fn is_static(&self) -> bool {
        self.inv_mass == 0.0
    }
    
    pub fn apply_impulse(&mut self, impulse: Vec3, r: Vec3) {
        self.velocity += impulse * self.inv_mass;
        let angular_impulse = r.cross(impulse);
        self.angular_velocity += angular_impulse * self.inv_inertia;
    }
    
    /// Apply impulse to pseudo-velocities (for split impulse position correction).
    pub fn apply_pseudo_impulse(&mut self, impulse: Vec3, r: Vec3) {
        self.pseudo_velocity += impulse * self.inv_mass;
        let angular_impulse = r.cross(impulse);
        self.pseudo_angular_velocity += angular_impulse * self.inv_inertia;
    }
    
    /// Apply pseudo-velocities to position (called after solving).
    pub fn apply_pseudo_positions(&mut self, dt: f32) {
        self.position += self.pseudo_velocity * dt;
        
        // Integrate pseudo-angular velocity into rotation
        if self.pseudo_angular_velocity.length_squared() > 1e-10 {
            let omega = self.pseudo_angular_velocity;
            let dq = Quat::from_xyzw(
                omega.x * 0.5 * dt,
                omega.y * 0.5 * dt,
                omega.z * 0.5 * dt,
                0.0,
            ) * self.rotation;
            self.rotation = (self.rotation + dq).normalize();
        }
    }
}

/// Sequential impulse solver
pub struct ConstraintSolver {
    config: SolverConfig,
}

impl ConstraintSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }
    
    pub fn with_default() -> Self {
        Self::new(SolverConfig::default())
    }
    
    /// Solve contact constraints
    pub fn solve_contacts(
        &self,
        contacts: &mut [ContactConstraint],
        bodies: &mut [SolverBody],
    ) {
        // Warm start
        if self.config.warm_starting {
            for contact in contacts.iter() {
                let impulses = contact.warm_start();
                for imp in impulses {
                    let body_a = &mut bodies[imp.body_a as usize];
                    body_a.apply_impulse(-imp.impulse, imp.r_a);
                    
                    let body_b = &mut bodies[imp.body_b as usize];
                    body_b.apply_impulse(imp.impulse, imp.r_b);
                }
            }
        }
        
        // Velocity iterations
        for _ in 0..self.config.velocity_iterations {
            for contact in contacts.iter_mut() {
                let body_a = bodies[contact.body_a as usize];
                let body_b = bodies[contact.body_b as usize];
                
                let impulses = contact.solve_velocity(
                    body_a.velocity,
                    body_a.angular_velocity,
                    body_b.velocity,
                    body_b.angular_velocity,
                );
                
                for imp in impulses {
                    let body_a = &mut bodies[imp.body_a as usize];
                    body_a.apply_impulse(-imp.impulse, imp.r_a);
                    
                    let body_b = &mut bodies[imp.body_b as usize];
                    body_b.apply_impulse(imp.impulse, imp.r_b);
                }
            }
        }
        
        // Position correction based on configured method
        match self.config.position_correction {
            PositionCorrection::Baumgarte => {
                // Position iterations using direct correction
                for _ in 0..self.config.position_iterations {
                    for contact in contacts.iter() {
                        let body_a = bodies[contact.body_a as usize];
                        let body_b = bodies[contact.body_b as usize];
                        
                        let correction = contact.solve_position(
                            body_a.position,
                            body_b.position,
                            body_a.inv_mass,
                            body_b.inv_mass,
                        );
                        
                        bodies[contact.body_a as usize].position += correction.correction_a;
                        bodies[contact.body_b as usize].position += correction.correction_b;
                    }
                }
            }
            PositionCorrection::SplitImpulse => {
                // Reset pseudo velocities
                for body in bodies.iter_mut() {
                    body.pseudo_velocity = Vec3::ZERO;
                    body.pseudo_angular_velocity = Vec3::ZERO;
                }
                
                // Solve position with pseudo-impulses
                for _ in 0..self.config.position_iterations {
                    for contact in contacts.iter() {
                        let body_a = bodies[contact.body_a as usize];
                        let body_b = bodies[contact.body_b as usize];
                        
                        // Compute position error
                        let penetration = contact.penetration;
                        
                        // Only correct if deeper than threshold
                        if penetration > self.config.split_impulse_threshold {
                            continue;
                        }
                        
                        let error = penetration - self.config.slop;
                        if error >= 0.0 {
                            continue;
                        }
                        
                        // Compute pseudo-impulse magnitude
                        let bias = -self.config.split_impulse_beta * error;
                        
                        // Get relative pseudo-velocity along normal
                        let r_a = contact.r_a;
                        let r_b = contact.r_b;
                        let normal = contact.normal;
                        
                        let v_a = body_a.pseudo_velocity + body_a.pseudo_angular_velocity.cross(r_a);
                        let v_b = body_b.pseudo_velocity + body_b.pseudo_angular_velocity.cross(r_b);
                        let rel_vel = v_b - v_a;
                        let normal_vel = rel_vel.dot(normal);
                        
                        // Compute impulse
                        let impulse_mag = (bias - normal_vel) * contact.normal_mass;
                        
                        // Clamp to non-negative (only push apart)
                        let impulse_mag = impulse_mag.max(0.0);
                        let impulse = normal * impulse_mag;
                        
                        // Apply to pseudo-velocities
                        bodies[contact.body_a as usize].apply_pseudo_impulse(-impulse, r_a);
                        bodies[contact.body_b as usize].apply_pseudo_impulse(impulse, r_b);
                    }
                }
            }
            PositionCorrection::NGS => {
                // Non-linear Gauss-Seidel: directly solve position constraints
                for _ in 0..self.config.position_iterations {
                    for contact in contacts.iter() {
                        let body_a = &bodies[contact.body_a as usize];
                        let body_b = &bodies[contact.body_b as usize];
                        
                        // Recompute penetration with current positions
                        let world_a = body_a.position + contact.r_a;
                        let world_b = body_b.position + contact.r_b;
                        let sep = (world_a - world_b).dot(contact.normal);
                        
                        let error = sep + self.config.slop;
                        if error >= 0.0 {
                            continue;
                        }
                        
                        // Compute correction magnitude
                        let total_inv_mass = body_a.inv_mass + body_b.inv_mass;
                        if total_inv_mass < 1e-10 {
                            continue;
                        }
                        
                        let correction = -error / total_inv_mass;
                        let correction_a = contact.normal * (-correction * body_a.inv_mass);
                        let correction_b = contact.normal * (correction * body_b.inv_mass);
                        
                        bodies[contact.body_a as usize].position += correction_a;
                        bodies[contact.body_b as usize].position += correction_b;
                    }
                }
            }
        }
    }
    
    /// Solve joint constraints with a handle-to-index mapping.
    /// The mapping function converts BodyHandle to array index.
    pub fn solve_joints<F>(
        &self,
        joints: &mut [Box<dyn Joint>],
        bodies: &mut [SolverBody],
        handle_to_index: F,
    )
    where
        F: Fn(BodyHandle) -> usize,
    {
        // Warm start
        if self.config.warm_starting {
            for joint in joints.iter() {
                let impulses = joint.warm_start();
                for imp in impulses {
                    let idx_a = handle_to_index(imp.body_a);
                    let idx_b = handle_to_index(imp.body_b);
                    
                    let body_a = &mut bodies[idx_a];
                    body_a.velocity -= imp.linear * body_a.inv_mass;
                    body_a.angular_velocity -= imp.angular_a;
                    
                    let body_b = &mut bodies[idx_b];
                    body_b.velocity += imp.linear * body_b.inv_mass;
                    body_b.angular_velocity += imp.angular_b;
                }
            }
        }
        
        // Velocity iterations
        for _ in 0..self.config.velocity_iterations {
            for joint in joints.iter_mut() {
                let (ha, hb) = joint.bodies();
                let idx_a = handle_to_index(ha);
                let idx_b = handle_to_index(hb);
                let body_a = bodies[idx_a];
                let body_b = bodies[idx_b];
                
                let impulses = joint.solve_velocity(
                    body_a.velocity,
                    body_a.angular_velocity,
                    body_b.velocity,
                    body_b.angular_velocity,
                    body_a.inv_mass,
                    body_b.inv_mass,
                    body_a.inv_inertia,
                    body_b.inv_inertia,
                );
                
                for imp in impulses {
                    let idx_a = handle_to_index(imp.body_a);
                    let idx_b = handle_to_index(imp.body_b);
                    
                    let body_a = &mut bodies[idx_a];
                    body_a.velocity -= imp.linear * body_a.inv_mass;
                    body_a.angular_velocity -= imp.angular_a;
                    
                    let body_b = &mut bodies[idx_b];
                    body_b.velocity += imp.linear * body_b.inv_mass;
                    body_b.angular_velocity += imp.angular_b;
                }
            }
        }
        
        // Position iterations
        for _ in 0..self.config.position_iterations {
            for joint in joints.iter_mut() {
                let (ha, hb) = joint.bodies();
                let idx_a = handle_to_index(ha);
                let idx_b = handle_to_index(hb);
                let body_a = bodies[idx_a];
                let body_b = bodies[idx_b];
                
                let result = joint.solve_position(
                    body_a.position,
                    body_a.rotation,
                    body_b.position,
                    body_b.rotation,
                    body_a.inv_mass,
                    body_b.inv_mass,
                );
                
                bodies[idx_a].position += result.delta_pos_a;
                bodies[idx_a].rotation = result.delta_rot_a * bodies[idx_a].rotation;
                bodies[idx_b].position += result.delta_pos_b;
                bodies[idx_b].rotation = result.delta_rot_b * bodies[idx_b].rotation;
            }
        }
    }
    
    /// Get config
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }
    
    /// Set velocity iterations
    pub fn set_velocity_iterations(&mut self, iterations: usize) {
        self.config.velocity_iterations = iterations;
    }
    
    /// Set position iterations
    pub fn set_position_iterations(&mut self, iterations: usize) {
        self.config.position_iterations = iterations;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::contact::{ContactManifold, ContactPoint};
    use slotmap::SlotMap;

    fn make_body(pos: Vec3, vel: Vec3) -> SolverBody {
        SolverBody::new(
            pos,
            Quat::IDENTITY,
            vel,
            Vec3::ZERO,
            1.0,
            Vec3::ONE,
        )
    }

    fn make_static_body(pos: Vec3) -> SolverBody {
        SolverBody::new(
            pos,
            Quat::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
            Vec3::ZERO,
        )
    }
    
    fn make_handles() -> (SlotMap<BodyHandle, ()>, BodyHandle, BodyHandle) {
        let mut map = SlotMap::with_key();
        let h1 = map.insert(());
        let h2 = map.insert(());
        (map, h1, h2)
    }

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.velocity_iterations, 8);
        assert_eq!(config.position_iterations, 3);
    }

    #[test]
    fn test_solver_body_is_static() {
        let dynamic = make_body(Vec3::ZERO, Vec3::ZERO);
        let static_body = make_static_body(Vec3::ZERO);
        
        assert!(!dynamic.is_static());
        assert!(static_body.is_static());
    }

    #[test]
    fn test_solver_body_apply_impulse() {
        let mut body = make_body(Vec3::ZERO, Vec3::ZERO);
        body.apply_impulse(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO);
        
        assert_eq!(body.velocity, Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn test_solver_new() {
        let solver = ConstraintSolver::with_default();
        assert_eq!(solver.config().velocity_iterations, 8);
    }

    #[test]
    fn test_solve_contacts() {
        let solver = ConstraintSolver::with_default();
        let (_map, h1, h2) = make_handles();
        
        // Body A falling onto stationary body B
        let mut manifold = ContactManifold::new(h1, h2);
        manifold.set_normal(Vec3::NEG_Y);
        manifold.add_point(ContactPoint::new(
            Vec3::new(0.0, 0.05, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::NEG_Y,
            0.05,
        ));
        
        let contact = ContactConstraint::from_manifold_with_indices(
            &manifold,
            0, 1, // indices
            Vec3::new(0.0, 0.1, 0.0),
            Vec3::ZERO,
            1.0, 1.0,
            Vec3::ONE, Vec3::ONE,
            0.5, 0.3
        );
        
        let mut bodies = vec![
            make_body(Vec3::new(0.0, 0.1, 0.0), Vec3::new(0.0, -10.0, 0.0)),
            make_body(Vec3::ZERO, Vec3::ZERO),
        ];
        
        solver.solve_contacts(&mut [contact], &mut bodies);
        
        // Body 0 should slow down (less negative velocity)
        assert!(bodies[0].velocity.y > -10.0);
    }

    #[test]
    fn test_solver_set_iterations() {
        let mut solver = ConstraintSolver::with_default();
        solver.set_velocity_iterations(10);
        solver.set_position_iterations(5);
        
        assert_eq!(solver.config().velocity_iterations, 10);
        assert_eq!(solver.config().position_iterations, 5);
    }

    #[test]
    fn test_static_body_not_moved() {
        let solver = ConstraintSolver::with_default();
        let (_map, h1, h2) = make_handles();
        
        let mut manifold = ContactManifold::new(h1, h2);
        manifold.set_normal(Vec3::Y);
        manifold.add_point(ContactPoint::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.1, 0.0),
            Vec3::Y,
            0.1,
        ));
        
        let contact = ContactConstraint::from_manifold_with_indices(
            &manifold,
            0, 1,
            Vec3::ZERO,
            Vec3::new(0.0, 0.1, 0.0),
            0.0, 1.0, // body 0 is static
            Vec3::ZERO, Vec3::ONE,
            0.5, 0.3
        );
        
        let mut bodies = vec![
            make_static_body(Vec3::ZERO),
            make_body(Vec3::new(0.0, 0.1, 0.0), Vec3::new(0.0, -10.0, 0.0)),
        ];
        
        solver.solve_contacts(&mut [contact], &mut bodies);
        
        // Static body should not move
        assert_eq!(bodies[0].velocity, Vec3::ZERO);
        assert_eq!(bodies[0].position, Vec3::ZERO);
    }
}
