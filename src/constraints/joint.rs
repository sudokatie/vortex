// Joint trait and common types

use glam::Vec3;

/// Joint connecting two bodies
pub trait Joint: Send + Sync {
    /// Get body handles
    fn bodies(&self) -> (u32, u32);
    
    /// Apply warm starting impulses
    fn warm_start(&self) -> Vec<JointImpulse>;
    
    /// Solve velocity constraints
    fn solve_velocity(
        &mut self,
        vel_a: Vec3,
        ang_vel_a: Vec3,
        vel_b: Vec3,
        ang_vel_b: Vec3,
        inv_mass_a: f32,
        inv_mass_b: f32,
        inv_inertia_a: Vec3,
        inv_inertia_b: Vec3,
    ) -> Vec<JointImpulse>;
    
    /// Solve position constraints
    fn solve_position(
        &mut self,
        pos_a: Vec3,
        rot_a: glam::Quat,
        pos_b: Vec3,
        rot_b: glam::Quat,
        inv_mass_a: f32,
        inv_mass_b: f32,
    ) -> PositionResult;
    
    /// Check if joint is broken
    fn is_broken(&self) -> bool {
        false
    }
    
    /// Get breaking force threshold
    fn breaking_force(&self) -> Option<f32> {
        None
    }
}

/// Impulse from joint solver
#[derive(Debug, Clone, Copy)]
pub struct JointImpulse {
    pub body_a: u32,
    pub body_b: u32,
    /// Linear impulse
    pub linear: Vec3,
    /// Angular impulse for body A
    pub angular_a: Vec3,
    /// Angular impulse for body B
    pub angular_b: Vec3,
}

impl JointImpulse {
    pub fn new(body_a: u32, body_b: u32, linear: Vec3) -> Self {
        Self {
            body_a,
            body_b,
            linear,
            angular_a: Vec3::ZERO,
            angular_b: Vec3::ZERO,
        }
    }
    
    pub fn with_angular(mut self, angular_a: Vec3, angular_b: Vec3) -> Self {
        self.angular_a = angular_a;
        self.angular_b = angular_b;
        self
    }
}

/// Position correction result
#[derive(Debug, Clone, Copy, Default)]
pub struct PositionResult {
    pub delta_pos_a: Vec3,
    pub delta_rot_a: glam::Quat,
    pub delta_pos_b: Vec3,
    pub delta_rot_b: glam::Quat,
}

impl PositionResult {
    pub fn position_only(delta_a: Vec3, delta_b: Vec3) -> Self {
        Self {
            delta_pos_a: delta_a,
            delta_rot_a: glam::Quat::IDENTITY,
            delta_pos_b: delta_b,
            delta_rot_b: glam::Quat::IDENTITY,
        }
    }
}

/// Joint limits
#[derive(Debug, Clone, Copy)]
pub struct JointLimit {
    pub lower: f32,
    pub upper: f32,
    pub enabled: bool,
}

impl JointLimit {
    pub fn new(lower: f32, upper: f32) -> Self {
        Self { lower, upper, enabled: true }
    }
    
    pub fn disabled() -> Self {
        Self { lower: 0.0, upper: 0.0, enabled: false }
    }
    
    pub fn symmetric(limit: f32) -> Self {
        Self::new(-limit, limit)
    }
    
    pub fn clamp(&self, value: f32) -> f32 {
        if self.enabled {
            value.clamp(self.lower, self.upper)
        } else {
            value
        }
    }
    
    pub fn is_at_limit(&self, value: f32) -> bool {
        self.enabled && (value <= self.lower || value >= self.upper)
    }
}

/// Joint motor
#[derive(Debug, Clone, Copy)]
pub struct JointMotor {
    pub target_velocity: f32,
    pub max_force: f32,
    pub enabled: bool,
}

impl JointMotor {
    pub fn new(target_velocity: f32, max_force: f32) -> Self {
        Self { target_velocity, max_force, enabled: true }
    }
    
    pub fn disabled() -> Self {
        Self { target_velocity: 0.0, max_force: 0.0, enabled: false }
    }
    
    /// Compute motor impulse
    pub fn compute_impulse(&self, current_velocity: f32, effective_mass: f32, dt: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        
        let velocity_error = self.target_velocity - current_velocity;
        let impulse = effective_mass * velocity_error;
        let max_impulse = self.max_force * dt;
        
        impulse.clamp(-max_impulse, max_impulse)
    }
}

/// Spring for soft constraints
#[derive(Debug, Clone, Copy)]
pub struct JointSpring {
    pub stiffness: f32,
    pub damping: f32,
    pub enabled: bool,
}

impl JointSpring {
    pub fn new(stiffness: f32, damping: f32) -> Self {
        Self { stiffness, damping, enabled: true }
    }
    
    pub fn disabled() -> Self {
        Self { stiffness: 0.0, damping: 0.0, enabled: false }
    }
    
    /// Compute spring force
    pub fn compute_force(&self, displacement: f32, velocity: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        
        -self.stiffness * displacement - self.damping * velocity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_joint_impulse_new() {
        let imp = JointImpulse::new(0, 1, Vec3::X);
        assert_eq!(imp.body_a, 0);
        assert_eq!(imp.body_b, 1);
        assert_eq!(imp.linear, Vec3::X);
    }

    #[test]
    fn test_joint_limit() {
        let limit = JointLimit::new(-1.0, 1.0);
        assert_eq!(limit.clamp(0.5), 0.5);
        assert_eq!(limit.clamp(2.0), 1.0);
        assert_eq!(limit.clamp(-2.0), -1.0);
    }

    #[test]
    fn test_joint_limit_disabled() {
        let limit = JointLimit::disabled();
        assert_eq!(limit.clamp(100.0), 100.0);
    }

    #[test]
    fn test_joint_limit_at_limit() {
        let limit = JointLimit::new(-1.0, 1.0);
        assert!(limit.is_at_limit(1.0));
        assert!(limit.is_at_limit(-1.0));
        assert!(!limit.is_at_limit(0.0));
    }

    #[test]
    fn test_joint_motor() {
        let motor = JointMotor::new(10.0, 100.0);
        let impulse = motor.compute_impulse(0.0, 1.0, 0.016);
        assert!(impulse > 0.0);
    }

    #[test]
    fn test_joint_motor_clamped() {
        let motor = JointMotor::new(1000.0, 10.0);
        let impulse = motor.compute_impulse(0.0, 1.0, 0.016);
        assert!(impulse <= 0.16 + 0.01);
    }

    #[test]
    fn test_joint_spring() {
        let spring = JointSpring::new(100.0, 10.0);
        let force = spring.compute_force(1.0, 0.0);
        assert_eq!(force, -100.0);
    }

    #[test]
    fn test_joint_spring_with_damping() {
        let spring = JointSpring::new(100.0, 10.0);
        let force = spring.compute_force(1.0, 1.0);
        assert_eq!(force, -110.0);
    }

    #[test]
    fn test_position_result_default() {
        let r = PositionResult::default();
        assert_eq!(r.delta_pos_a, Vec3::ZERO);
    }
}
