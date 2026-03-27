// Distance joint - maintains fixed distance between two anchor points

use glam::{Quat, Vec3};
use super::joint::{Joint, JointImpulse, JointLimit, JointSpring, PositionResult};

/// Distance joint - keeps two points at a fixed distance
#[derive(Debug, Clone)]
pub struct DistanceJoint {
    body_a: u32,
    body_b: u32,
    /// Local anchor on body A
    local_anchor_a: Vec3,
    /// Local anchor on body B
    local_anchor_b: Vec3,
    /// Target distance
    rest_length: f32,
    /// Distance limits
    limit: JointLimit,
    /// Spring for soft constraint
    spring: JointSpring,
    /// Accumulated impulse
    impulse: f32,
    /// Effective mass
    mass: f32,
    /// Current direction (A to B)
    direction: Vec3,
}

impl DistanceJoint {
    pub fn new(
        body_a: u32,
        body_b: u32,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        rest_length: f32,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a,
            local_anchor_b,
            rest_length,
            limit: JointLimit::disabled(),
            spring: JointSpring::disabled(),
            impulse: 0.0,
            mass: 0.0,
            direction: Vec3::X,
        }
    }
    
    pub fn with_limit(mut self, min_length: f32, max_length: f32) -> Self {
        self.limit = JointLimit::new(min_length, max_length);
        self
    }
    
    pub fn with_spring(mut self, stiffness: f32, damping: f32) -> Self {
        self.spring = JointSpring::new(stiffness, damping);
        self
    }
    
    /// Initialize constraint for solving
    pub fn prepare(
        &mut self,
        pos_a: Vec3,
        rot_a: Quat,
        pos_b: Vec3,
        rot_b: Quat,
        inv_mass_a: f32,
        inv_mass_b: f32,
    ) {
        let world_anchor_a = pos_a + rot_a * self.local_anchor_a;
        let world_anchor_b = pos_b + rot_b * self.local_anchor_b;
        
        let delta = world_anchor_b - world_anchor_a;
        let length = delta.length();
        
        if length > 1e-6 {
            self.direction = delta / length;
        } else {
            self.direction = Vec3::X;
        }
        
        self.mass = inv_mass_a + inv_mass_b;
        if self.mass > 0.0 {
            self.mass = 1.0 / self.mass;
        }
    }
    
    /// Get current distance
    pub fn current_length(&self, pos_a: Vec3, rot_a: Quat, pos_b: Vec3, rot_b: Quat) -> f32 {
        let world_anchor_a = pos_a + rot_a * self.local_anchor_a;
        let world_anchor_b = pos_b + rot_b * self.local_anchor_b;
        (world_anchor_b - world_anchor_a).length()
    }
}

impl Joint for DistanceJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
    
    fn warm_start(&self) -> Vec<JointImpulse> {
        if self.impulse.abs() < 1e-6 {
            return Vec::new();
        }
        
        let linear = self.direction * self.impulse;
        vec![JointImpulse::new(self.body_a, self.body_b, linear)]
    }
    
    fn solve_velocity(
        &mut self,
        vel_a: Vec3,
        _ang_vel_a: Vec3,
        vel_b: Vec3,
        _ang_vel_b: Vec3,
        _inv_mass_a: f32,
        _inv_mass_b: f32,
        _inv_inertia_a: Vec3,
        _inv_inertia_b: Vec3,
    ) -> Vec<JointImpulse> {
        // Relative velocity along constraint axis
        let rel_vel = (vel_b - vel_a).dot(self.direction);
        
        // Compute impulse
        let lambda = -self.mass * rel_vel;
        self.impulse += lambda;
        
        let linear = self.direction * lambda;
        vec![JointImpulse::new(self.body_a, self.body_b, linear)]
    }
    
    fn solve_position(
        &mut self,
        pos_a: Vec3,
        rot_a: Quat,
        pos_b: Vec3,
        rot_b: Quat,
        inv_mass_a: f32,
        inv_mass_b: f32,
    ) -> PositionResult {
        let world_anchor_a = pos_a + rot_a * self.local_anchor_a;
        let world_anchor_b = pos_b + rot_b * self.local_anchor_b;
        
        let delta = world_anchor_b - world_anchor_a;
        let length = delta.length();
        
        if length < 1e-6 {
            return PositionResult::default();
        }
        
        let direction = delta / length;
        let error = if self.limit.enabled {
            let clamped = self.limit.clamp(length);
            length - clamped
        } else {
            length - self.rest_length
        };
        
        if error.abs() < 0.001 {
            return PositionResult::default();
        }
        
        let total_inv_mass = inv_mass_a + inv_mass_b;
        if total_inv_mass < 1e-6 {
            return PositionResult::default();
        }
        
        let correction = error * 0.2; // Baumgarte factor
        let delta_a = direction * correction * (inv_mass_a / total_inv_mass);
        let delta_b = -direction * correction * (inv_mass_b / total_inv_mass);
        
        PositionResult::position_only(delta_a, delta_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_joint_new() {
        let j = DistanceJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0);
        assert_eq!(j.bodies(), (0, 1));
        assert_eq!(j.rest_length, 1.0);
    }

    #[test]
    fn test_distance_joint_with_limit() {
        let j = DistanceJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0)
            .with_limit(0.5, 1.5);
        assert!(j.limit.enabled);
    }

    #[test]
    fn test_distance_joint_prepare() {
        let mut j = DistanceJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0);
        j.prepare(
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY,
            1.0, 1.0
        );
        assert!((j.direction - Vec3::X).length() < 0.01);
    }

    #[test]
    fn test_distance_joint_current_length() {
        let j = DistanceJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0);
        let len = j.current_length(
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::new(3.0, 4.0, 0.0), Quat::IDENTITY
        );
        assert!((len - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_joint_solve_velocity() {
        let mut j = DistanceJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0);
        j.prepare(
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::X, Quat::IDENTITY,
            1.0, 1.0
        );
        
        let impulses = j.solve_velocity(
            Vec3::ZERO, Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO,
            1.0, 1.0, Vec3::ONE, Vec3::ONE
        );
        
        assert_eq!(impulses.len(), 1);
        assert!(impulses[0].linear.x < 0.0); // Should push B back
    }

    #[test]
    fn test_distance_joint_solve_position() {
        let mut j = DistanceJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0);
        let result = j.solve_position(
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY,
            1.0, 1.0
        );
        
        // Distance is 2, rest is 1, so should pull together
        // A moves toward +x, B moves toward -x
        assert!(result.delta_pos_a.x > 0.0 || result.delta_pos_b.x < 0.0);
    }
}
