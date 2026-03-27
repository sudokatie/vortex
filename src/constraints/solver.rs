// Sequential impulse constraint solver

use glam::{Quat, Vec3};
use super::contact::ContactConstraint;
use super::joint::Joint;

/// Solver configuration
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Number of velocity iterations
    pub velocity_iterations: usize,
    /// Number of position iterations
    pub position_iterations: usize,
    /// Baumgarte stabilization factor
    pub baumgarte: f32,
    /// Slop for penetration allowance
    pub slop: f32,
    /// Enable warm starting
    pub warm_starting: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            velocity_iterations: 8,
            position_iterations: 3,
            baumgarte: 0.2,
            slop: 0.005,
            warm_starting: true,
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
        
        // Position iterations
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
    
    /// Solve joint constraints
    pub fn solve_joints(
        &self,
        joints: &mut [Box<dyn Joint>],
        bodies: &mut [SolverBody],
    ) {
        // Warm start
        if self.config.warm_starting {
            for joint in joints.iter() {
                let impulses = joint.warm_start();
                for imp in impulses {
                    let body_a = &mut bodies[imp.body_a as usize];
                    body_a.velocity -= imp.linear * body_a.inv_mass;
                    body_a.angular_velocity -= imp.angular_a;
                    
                    let body_b = &mut bodies[imp.body_b as usize];
                    body_b.velocity += imp.linear * body_b.inv_mass;
                    body_b.angular_velocity += imp.angular_b;
                }
            }
        }
        
        // Velocity iterations
        for _ in 0..self.config.velocity_iterations {
            for joint in joints.iter_mut() {
                let (a, b) = joint.bodies();
                let body_a = bodies[a as usize];
                let body_b = bodies[b as usize];
                
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
                    let body_a = &mut bodies[imp.body_a as usize];
                    body_a.velocity -= imp.linear * body_a.inv_mass;
                    body_a.angular_velocity -= imp.angular_a;
                    
                    let body_b = &mut bodies[imp.body_b as usize];
                    body_b.velocity += imp.linear * body_b.inv_mass;
                    body_b.angular_velocity += imp.angular_b;
                }
            }
        }
        
        // Position iterations
        for _ in 0..self.config.position_iterations {
            for joint in joints.iter_mut() {
                let (a, b) = joint.bodies();
                let body_a = bodies[a as usize];
                let body_b = bodies[b as usize];
                
                let result = joint.solve_position(
                    body_a.position,
                    body_a.rotation,
                    body_b.position,
                    body_b.rotation,
                    body_a.inv_mass,
                    body_b.inv_mass,
                );
                
                bodies[a as usize].position += result.delta_pos_a;
                bodies[a as usize].rotation = result.delta_rot_a * bodies[a as usize].rotation;
                bodies[b as usize].position += result.delta_pos_b;
                bodies[b as usize].rotation = result.delta_rot_b * bodies[b as usize].rotation;
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
        
        // Body A falling onto stationary body B
        // A is at y=0.1, falling at -10
        // B is at y=0, stationary (floor)
        // Contact normal points from A to B (-Y, downward)
        let mut manifold = ContactManifold::new(0, 1);
        manifold.add_point(ContactPoint::new(
            Vec3::new(0.0, 0.05, 0.0),  // point on A
            Vec3::new(0.0, 0.0, 0.0),   // point on B
            Vec3::NEG_Y,                 // normal from A to B (down)
            0.05,                        // penetration depth
        ));
        
        let mut contact = ContactConstraint::from_manifold(
            &manifold,
            Vec3::new(0.0, 0.1, 0.0),   // pos_a (above)
            Vec3::ZERO,                  // pos_b (floor)
            1.0, 1.0,
            Vec3::ONE, Vec3::ONE,
            0.5, 0.3
        );
        
        let mut bodies = vec![
            make_body(Vec3::new(0.0, 0.1, 0.0), Vec3::new(0.0, -10.0, 0.0)),  // A: falling
            make_body(Vec3::ZERO, Vec3::ZERO),                                 // B: floor
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
        
        let mut manifold = ContactManifold::new(0, 1);
        manifold.add_point(ContactPoint::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.1, 0.0),
            Vec3::Y,
            0.1,
        ));
        
        let mut contact = ContactConstraint::from_manifold(
            &manifold,
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
