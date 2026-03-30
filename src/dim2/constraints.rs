//! 2D constraint solver and joints.

use glam::Vec2;

use super::RigidBody2D;

// =============================================================================
// Solver Body
// =============================================================================

/// 2D body state for solver.
#[derive(Debug, Clone, Copy)]
pub struct SolverBody2D {
    pub position: Vec2,
    pub rotation: f32,
    pub velocity: Vec2,
    pub angular_velocity: f32,
    pub inv_mass: f32,
    pub inv_inertia: f32,
}

impl SolverBody2D {
    pub fn from_body(body: &RigidBody2D) -> Self {
        Self {
            position: body.position,
            rotation: body.rotation,
            velocity: body.velocity,
            angular_velocity: body.angular_velocity,
            inv_mass: body.inv_mass,
            inv_inertia: body.inv_inertia,
        }
    }
    
    pub fn is_static(&self) -> bool {
        self.inv_mass == 0.0
    }
    
    pub fn apply_impulse(&mut self, impulse: Vec2, r: Vec2) {
        self.velocity += impulse * self.inv_mass;
        self.angular_velocity += r.perp_dot(impulse) * self.inv_inertia;
    }
}

// =============================================================================
// Contact Constraint
// =============================================================================

/// 2D contact constraint.
#[derive(Debug, Clone)]
pub struct ContactConstraint2D {
    pub body_a: usize,
    pub body_b: usize,
    pub normal: Vec2,
    pub r_a: Vec2,
    pub r_b: Vec2,
    pub penetration: f32,
    pub normal_mass: f32,
    pub tangent_mass: f32,
    pub normal_impulse: f32,
    pub tangent_impulse: f32,
    pub friction: f32,
    pub restitution: f32,
    pub bias: f32,
}

impl ContactConstraint2D {
    pub fn new(
        body_a: usize,
        body_b: usize,
        normal: Vec2,
        r_a: Vec2,
        r_b: Vec2,
        penetration: f32,
        inv_mass_a: f32,
        inv_mass_b: f32,
        inv_inertia_a: f32,
        inv_inertia_b: f32,
        friction: f32,
        restitution: f32,
    ) -> Self {
        let tangent = Vec2::new(-normal.y, normal.x);
        
        // Effective mass for normal
        let rn_a = r_a.perp_dot(normal);
        let rn_b = r_b.perp_dot(normal);
        let normal_mass = inv_mass_a + inv_mass_b 
            + inv_inertia_a * rn_a * rn_a 
            + inv_inertia_b * rn_b * rn_b;
        
        // Effective mass for tangent
        let rt_a = r_a.perp_dot(tangent);
        let rt_b = r_b.perp_dot(tangent);
        let tangent_mass = inv_mass_a + inv_mass_b 
            + inv_inertia_a * rt_a * rt_a 
            + inv_inertia_b * rt_b * rt_b;
        
        Self {
            body_a,
            body_b,
            normal,
            r_a,
            r_b,
            penetration,
            normal_mass: if normal_mass > 0.0 { 1.0 / normal_mass } else { 0.0 },
            tangent_mass: if tangent_mass > 0.0 { 1.0 / tangent_mass } else { 0.0 },
            normal_impulse: 0.0,
            tangent_impulse: 0.0,
            friction,
            restitution,
            bias: 0.0,
        }
    }
    
    /// Solve velocity constraint.
    pub fn solve_velocity(&mut self, bodies: &mut [SolverBody2D]) {
        let body_a = bodies[self.body_a];
        let body_b = bodies[self.body_b];
        
        let tangent = Vec2::new(-self.normal.y, self.normal.x);
        
        // Relative velocity at contact
        let vel_a = body_a.velocity + Vec2::new(-self.r_a.y, self.r_a.x) * body_a.angular_velocity;
        let vel_b = body_b.velocity + Vec2::new(-self.r_b.y, self.r_b.x) * body_b.angular_velocity;
        let rel_vel = vel_b - vel_a;
        
        // Normal impulse
        let vn = rel_vel.dot(self.normal);
        let lambda_n = self.normal_mass * (-vn + self.bias);
        
        let old_impulse = self.normal_impulse;
        self.normal_impulse = (self.normal_impulse + lambda_n).max(0.0);
        let applied_n = self.normal_impulse - old_impulse;
        
        let impulse_n = self.normal * applied_n;
        bodies[self.body_a].apply_impulse(-impulse_n, self.r_a);
        bodies[self.body_b].apply_impulse(impulse_n, self.r_b);
        
        // Tangent (friction) impulse
        let vel_a = bodies[self.body_a].velocity + Vec2::new(-self.r_a.y, self.r_a.x) * bodies[self.body_a].angular_velocity;
        let vel_b = bodies[self.body_b].velocity + Vec2::new(-self.r_b.y, self.r_b.x) * bodies[self.body_b].angular_velocity;
        let rel_vel = vel_b - vel_a;
        let vt = rel_vel.dot(tangent);
        
        let lambda_t = self.tangent_mass * (-vt);
        let max_friction = self.friction * self.normal_impulse;
        
        let old_impulse = self.tangent_impulse;
        self.tangent_impulse = (self.tangent_impulse + lambda_t).clamp(-max_friction, max_friction);
        let applied_t = self.tangent_impulse - old_impulse;
        
        let impulse_t = tangent * applied_t;
        bodies[self.body_a].apply_impulse(-impulse_t, self.r_a);
        bodies[self.body_b].apply_impulse(impulse_t, self.r_b);
    }
}

// =============================================================================
// Distance Joint
// =============================================================================

/// 2D distance joint - maintains fixed distance between two points.
#[derive(Debug, Clone)]
pub struct DistanceJoint2D {
    pub body_a: usize,
    pub body_b: usize,
    pub local_anchor_a: Vec2,
    pub local_anchor_b: Vec2,
    pub target_distance: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub impulse: f32,
}

impl DistanceJoint2D {
    pub fn new(
        body_a: usize,
        body_b: usize,
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
        target_distance: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a,
            local_anchor_b,
            target_distance,
            stiffness: 0.0, // 0 = rigid
            damping: 0.0,
            impulse: 0.0,
        }
    }
    
    pub fn with_soft(mut self, stiffness: f32, damping: f32) -> Self {
        self.stiffness = stiffness;
        self.damping = damping;
        self
    }
    
    /// Solve velocity constraint.
    pub fn solve_velocity(&mut self, bodies: &mut [SolverBody2D], dt: f32) {
        let body_a = &bodies[self.body_a];
        let body_b = &bodies[self.body_b];
        
        // World anchors
        let cos_a = body_a.rotation.cos();
        let sin_a = body_a.rotation.sin();
        let r_a = Vec2::new(
            self.local_anchor_a.x * cos_a - self.local_anchor_a.y * sin_a,
            self.local_anchor_a.x * sin_a + self.local_anchor_a.y * cos_a,
        );
        
        let cos_b = body_b.rotation.cos();
        let sin_b = body_b.rotation.sin();
        let r_b = Vec2::new(
            self.local_anchor_b.x * cos_b - self.local_anchor_b.y * sin_b,
            self.local_anchor_b.x * sin_b + self.local_anchor_b.y * cos_b,
        );
        
        let world_a = body_a.position + r_a;
        let world_b = body_b.position + r_b;
        
        let d = world_b - world_a;
        let current_dist = d.length();
        
        if current_dist < 1e-6 {
            return;
        }
        
        let axis = d / current_dist;
        
        // Relative velocity along axis
        let vel_a = body_a.velocity + Vec2::new(-r_a.y, r_a.x) * body_a.angular_velocity;
        let vel_b = body_b.velocity + Vec2::new(-r_b.y, r_b.x) * body_b.angular_velocity;
        let rel_vel = (vel_b - vel_a).dot(axis);
        
        // Effective mass
        let rn_a = r_a.perp_dot(axis);
        let rn_b = r_b.perp_dot(axis);
        let eff_mass = body_a.inv_mass + body_b.inv_mass 
            + body_a.inv_inertia * rn_a * rn_a 
            + body_b.inv_inertia * rn_b * rn_b;
        
        if eff_mass < 1e-10 {
            return;
        }
        
        let mass = 1.0 / eff_mass;
        
        // Compute impulse
        let c = current_dist - self.target_distance;
        let bias = if self.stiffness > 0.0 {
            // Soft constraint
            let omega = (self.stiffness / mass).sqrt();
            let d = 2.0 * mass * self.damping * omega;
            let k = mass * omega * omega;
            let gamma = 1.0 / (dt * (d + dt * k));
            gamma * c
        } else {
            // Rigid constraint
            0.2 * c / dt
        };
        
        let lambda = -mass * (rel_vel + bias);
        self.impulse += lambda;
        
        let impulse = axis * lambda;
        bodies[self.body_a].apply_impulse(-impulse, r_a);
        bodies[self.body_b].apply_impulse(impulse, r_b);
    }
}

// =============================================================================
// Revolute Joint
// =============================================================================

/// 2D revolute joint (hinge) - allows rotation around a point.
#[derive(Debug, Clone)]
pub struct RevoluteJoint2D {
    pub body_a: usize,
    pub body_b: usize,
    pub local_anchor_a: Vec2,
    pub local_anchor_b: Vec2,
    pub reference_angle: f32,
    pub lower_angle: f32,
    pub upper_angle: f32,
    pub limit_enabled: bool,
    pub motor_speed: f32,
    pub max_motor_torque: f32,
    pub motor_enabled: bool,
    pub impulse: Vec2,
    pub motor_impulse: f32,
    pub limit_impulse: f32,
}

impl RevoluteJoint2D {
    pub fn new(
        body_a: usize,
        body_b: usize,
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a,
            local_anchor_b,
            reference_angle: 0.0,
            lower_angle: 0.0,
            upper_angle: 0.0,
            limit_enabled: false,
            motor_speed: 0.0,
            max_motor_torque: 0.0,
            motor_enabled: false,
            impulse: Vec2::ZERO,
            motor_impulse: 0.0,
            limit_impulse: 0.0,
        }
    }
    
    pub fn with_reference_angle(mut self, angle: f32) -> Self {
        self.reference_angle = angle;
        self
    }
    
    pub fn with_limits(mut self, lower: f32, upper: f32) -> Self {
        self.lower_angle = lower;
        self.upper_angle = upper;
        self.limit_enabled = true;
        self
    }
    
    pub fn with_motor(mut self, speed: f32, max_torque: f32) -> Self {
        self.motor_speed = speed;
        self.max_motor_torque = max_torque;
        self.motor_enabled = true;
        self
    }
    
    /// Get current joint angle.
    pub fn angle(&self, bodies: &[SolverBody2D]) -> f32 {
        let body_a = &bodies[self.body_a];
        let body_b = &bodies[self.body_b];
        body_b.rotation - body_a.rotation - self.reference_angle
    }
    
    /// Solve velocity constraint.
    pub fn solve_velocity(&mut self, bodies: &mut [SolverBody2D]) {
        let body_a = bodies[self.body_a];
        let body_b = bodies[self.body_b];
        
        // World anchors
        let cos_a = body_a.rotation.cos();
        let sin_a = body_a.rotation.sin();
        let r_a = Vec2::new(
            self.local_anchor_a.x * cos_a - self.local_anchor_a.y * sin_a,
            self.local_anchor_a.x * sin_a + self.local_anchor_a.y * cos_a,
        );
        
        let cos_b = body_b.rotation.cos();
        let sin_b = body_b.rotation.sin();
        let r_b = Vec2::new(
            self.local_anchor_b.x * cos_b - self.local_anchor_b.y * sin_b,
            self.local_anchor_b.x * sin_b + self.local_anchor_b.y * cos_b,
        );
        
        // Point-to-point constraint
        let vel_a = body_a.velocity + Vec2::new(-r_a.y, r_a.x) * body_a.angular_velocity;
        let vel_b = body_b.velocity + Vec2::new(-r_b.y, r_b.x) * body_b.angular_velocity;
        let rel_vel = vel_b - vel_a;
        
        // Effective mass matrix (simplified to scalar for each axis)
        let k = body_a.inv_mass + body_b.inv_mass;
        let mass = if k > 1e-10 { 1.0 / k } else { 0.0 };
        
        let lambda = -rel_vel * mass;
        self.impulse += lambda;
        
        bodies[self.body_a].apply_impulse(-lambda, r_a);
        bodies[self.body_b].apply_impulse(lambda, r_b);
        
        // Motor
        if self.motor_enabled {
            let rel_ang_vel = bodies[self.body_b].angular_velocity - bodies[self.body_a].angular_velocity;
            let vel_error = self.motor_speed - rel_ang_vel;
            
            let ang_mass = body_a.inv_inertia + body_b.inv_inertia;
            let motor_mass = if ang_mass > 1e-10 { 1.0 / ang_mass } else { 0.0 };
            
            let motor_lambda = motor_mass * vel_error;
            let old = self.motor_impulse;
            self.motor_impulse = (self.motor_impulse + motor_lambda)
                .clamp(-self.max_motor_torque, self.max_motor_torque);
            let applied = self.motor_impulse - old;
            
            bodies[self.body_a].angular_velocity -= applied * body_a.inv_inertia;
            bodies[self.body_b].angular_velocity += applied * body_b.inv_inertia;
        }
        
        // Limits
        if self.limit_enabled {
            let angle = bodies[self.body_b].rotation - bodies[self.body_a].rotation - self.reference_angle;
            let rel_ang_vel = bodies[self.body_b].angular_velocity - bodies[self.body_a].angular_velocity;
            
            let ang_mass = body_a.inv_inertia + body_b.inv_inertia;
            let limit_mass = if ang_mass > 1e-10 { 1.0 / ang_mass } else { 0.0 };
            
            if angle <= self.lower_angle && rel_ang_vel < 0.0 {
                let limit_lambda = -rel_ang_vel * limit_mass;
                let old = self.limit_impulse;
                self.limit_impulse = (self.limit_impulse + limit_lambda).max(0.0);
                let applied = self.limit_impulse - old;
                
                bodies[self.body_a].angular_velocity -= applied * body_a.inv_inertia;
                bodies[self.body_b].angular_velocity += applied * body_b.inv_inertia;
            } else if angle >= self.upper_angle && rel_ang_vel > 0.0 {
                let limit_lambda = -rel_ang_vel * limit_mass;
                let old = self.limit_impulse;
                self.limit_impulse = (self.limit_impulse + limit_lambda).min(0.0);
                let applied = self.limit_impulse - old;
                
                bodies[self.body_a].angular_velocity -= applied * body_a.inv_inertia;
                bodies[self.body_b].angular_velocity += applied * body_b.inv_inertia;
            }
        }
    }
}

// =============================================================================
// Constraint Solver
// =============================================================================

/// 2D constraint solver configuration.
#[derive(Debug, Clone)]
pub struct SolverConfig2D {
    pub velocity_iterations: usize,
    pub position_iterations: usize,
    pub baumgarte: f32,
    pub slop: f32,
}

impl Default for SolverConfig2D {
    fn default() -> Self {
        Self {
            velocity_iterations: 8,
            position_iterations: 3,
            baumgarte: 0.2,
            slop: 0.005,
        }
    }
}

/// 2D constraint solver.
pub struct ConstraintSolver2D {
    pub config: SolverConfig2D,
}

impl ConstraintSolver2D {
    pub fn new(config: SolverConfig2D) -> Self {
        Self { config }
    }
    
    /// Solve contacts and joints.
    pub fn solve(
        &self,
        contacts: &mut [ContactConstraint2D],
        distance_joints: &mut [DistanceJoint2D],
        revolute_joints: &mut [RevoluteJoint2D],
        bodies: &mut [SolverBody2D],
        dt: f32,
    ) {
        // Compute bias for contacts
        for contact in contacts.iter_mut() {
            let penetration = contact.penetration;
            if penetration > self.config.slop {
                contact.bias = self.config.baumgarte * (penetration - self.config.slop) / dt;
            } else {
                contact.bias = 0.0;
            }
        }
        
        // Velocity iterations
        for _ in 0..self.config.velocity_iterations {
            for contact in contacts.iter_mut() {
                contact.solve_velocity(bodies);
            }
            for joint in distance_joints.iter_mut() {
                joint.solve_velocity(bodies, dt);
            }
            for joint in revolute_joints.iter_mut() {
                joint.solve_velocity(bodies);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_solver_body_2d() {
        let mut body = SolverBody2D {
            position: Vec2::ZERO,
            rotation: 0.0,
            velocity: Vec2::ZERO,
            angular_velocity: 0.0,
            inv_mass: 1.0,
            inv_inertia: 1.0,
        };
        
        body.apply_impulse(Vec2::X, Vec2::Y);
        assert!(body.velocity.x > 0.0);
    }
    
    #[test]
    fn test_contact_constraint_2d() {
        let mut contact = ContactConstraint2D::new(
            0, 1,
            Vec2::Y,
            Vec2::ZERO, Vec2::ZERO,
            0.1,
            1.0, 1.0, 1.0, 1.0,
            0.5, 0.3
        );
        
        let mut bodies = vec![
            SolverBody2D {
                position: Vec2::ZERO,
                rotation: 0.0,
                velocity: Vec2::new(0.0, 1.0),
                angular_velocity: 0.0,
                inv_mass: 1.0,
                inv_inertia: 1.0,
            },
            SolverBody2D {
                position: Vec2::new(0.0, 1.0),
                rotation: 0.0,
                velocity: Vec2::new(0.0, -1.0),
                angular_velocity: 0.0,
                inv_mass: 1.0,
                inv_inertia: 1.0,
            },
        ];
        
        contact.solve_velocity(&mut bodies);
        
        // Velocities should be reduced
        assert!(bodies[0].velocity.y < 1.0);
        assert!(bodies[1].velocity.y > -1.0);
    }
    
    #[test]
    fn test_distance_joint_2d() {
        let mut joint = DistanceJoint2D::new(
            0, 1,
            Vec2::ZERO, Vec2::ZERO,
            2.0
        );
        
        let mut bodies = vec![
            SolverBody2D {
                position: Vec2::ZERO,
                rotation: 0.0,
                velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                inv_mass: 1.0,
                inv_inertia: 1.0,
            },
            SolverBody2D {
                position: Vec2::new(3.0, 0.0), // 3 units apart, target is 2
                rotation: 0.0,
                velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                inv_mass: 1.0,
                inv_inertia: 1.0,
            },
        ];
        
        joint.solve_velocity(&mut bodies, 1.0 / 60.0);
        
        // Bodies should be pushed toward each other
        assert!(bodies[0].velocity.x > 0.0);
        assert!(bodies[1].velocity.x < 0.0);
    }
    
    #[test]
    fn test_revolute_joint_2d() {
        let mut joint = RevoluteJoint2D::new(
            0, 1,
            Vec2::X, Vec2::new(-1.0, 0.0)
        );
        
        let mut bodies = vec![
            SolverBody2D {
                position: Vec2::ZERO,
                rotation: 0.0,
                velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                inv_mass: 1.0,
                inv_inertia: 1.0,
            },
            SolverBody2D {
                position: Vec2::new(2.0, 0.0),
                rotation: 0.0,
                velocity: Vec2::new(0.0, 1.0), // Moving away
                angular_velocity: 0.0,
                inv_mass: 1.0,
                inv_inertia: 1.0,
            },
        ];
        
        joint.solve_velocity(&mut bodies);
        
        // Joint should constrain the relative velocity
        assert!(bodies[1].velocity.y < 1.0);
    }
}
