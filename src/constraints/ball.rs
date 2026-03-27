// Ball joint (spherical joint) - allows rotation around a common point

use glam::{Quat, Vec3};
use super::joint::{Joint, JointImpulse, PositionResult};

/// Ball joint - point-to-point constraint allowing free rotation
#[derive(Debug, Clone)]
pub struct BallJoint {
    body_a: u32,
    body_b: u32,
    /// Local anchor on body A
    local_anchor_a: Vec3,
    /// Local anchor on body B
    local_anchor_b: Vec3,
    /// Accumulated impulse
    impulse: Vec3,
    /// Effective mass matrix (simplified as diagonal)
    mass: f32,
    /// World-space position of anchor A
    world_anchor_a: Vec3,
    /// World-space position of anchor B
    world_anchor_b: Vec3,
}

impl BallJoint {
    pub fn new(
        body_a: u32,
        body_b: u32,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a,
            local_anchor_b,
            impulse: Vec3::ZERO,
            mass: 0.0,
            world_anchor_a: Vec3::ZERO,
            world_anchor_b: Vec3::ZERO,
        }
    }
    
    /// Create ball joint from world-space anchor point
    pub fn from_world_anchor(
        body_a: u32,
        body_b: u32,
        pos_a: Vec3,
        rot_a: Quat,
        pos_b: Vec3,
        rot_b: Quat,
        world_anchor: Vec3,
    ) -> Self {
        let local_anchor_a = rot_a.inverse() * (world_anchor - pos_a);
        let local_anchor_b = rot_b.inverse() * (world_anchor - pos_b);
        Self::new(body_a, body_b, local_anchor_a, local_anchor_b)
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
        self.world_anchor_a = pos_a + rot_a * self.local_anchor_a;
        self.world_anchor_b = pos_b + rot_b * self.local_anchor_b;
        
        // Simplified effective mass (ignoring inertia for now)
        let total = inv_mass_a + inv_mass_b;
        self.mass = if total > 0.0 { 1.0 / total } else { 0.0 };
    }
    
    /// Get anchor separation
    pub fn separation(&self) -> Vec3 {
        self.world_anchor_b - self.world_anchor_a
    }
}

impl Joint for BallJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
    
    fn warm_start(&self) -> Vec<JointImpulse> {
        if self.impulse.length_squared() < 1e-12 {
            return Vec::new();
        }
        
        vec![JointImpulse::new(self.body_a, self.body_b, self.impulse)]
    }
    
    fn solve_velocity(
        &mut self,
        vel_a: Vec3,
        ang_vel_a: Vec3,
        vel_b: Vec3,
        ang_vel_b: Vec3,
        _inv_mass_a: f32,
        _inv_mass_b: f32,
        _inv_inertia_a: Vec3,
        _inv_inertia_b: Vec3,
    ) -> Vec<JointImpulse> {
        // Compute velocity at anchor points
        let r_a = self.world_anchor_a - vel_a; // Using vel_a as position proxy (simplified)
        let r_b = self.world_anchor_b - vel_b;
        
        let vel_a_at_anchor = vel_a + ang_vel_a.cross(r_a);
        let vel_b_at_anchor = vel_b + ang_vel_b.cross(r_b);
        
        // Relative velocity
        let rel_vel = vel_b_at_anchor - vel_a_at_anchor;
        
        // Compute impulse to eliminate relative velocity
        let lambda = -self.mass * rel_vel;
        self.impulse += lambda;
        
        vec![JointImpulse::new(self.body_a, self.body_b, lambda)]
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
        
        let error = world_anchor_b - world_anchor_a;
        let error_len = error.length();
        
        if error_len < 0.001 {
            return PositionResult::default();
        }
        
        let total_inv_mass = inv_mass_a + inv_mass_b;
        if total_inv_mass < 1e-6 {
            return PositionResult::default();
        }
        
        let correction = error * 0.2; // Baumgarte factor
        let delta_a = correction * (inv_mass_a / total_inv_mass);
        let delta_b = -correction * (inv_mass_b / total_inv_mass);
        
        PositionResult::position_only(delta_a, delta_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ball_joint_new() {
        let j = BallJoint::new(0, 1, Vec3::X, Vec3::NEG_X);
        assert_eq!(j.bodies(), (0, 1));
    }

    #[test]
    fn test_ball_joint_from_world() {
        let j = BallJoint::from_world_anchor(
            0, 1,
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY,
            Vec3::new(1.0, 0.0, 0.0)
        );
        assert!((j.local_anchor_a - Vec3::X).length() < 0.01);
        assert!((j.local_anchor_b - Vec3::NEG_X).length() < 0.01);
    }

    #[test]
    fn test_ball_joint_prepare() {
        let mut j = BallJoint::new(0, 1, Vec3::X, Vec3::NEG_X);
        j.prepare(
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY,
            1.0, 1.0
        );
        assert!(j.mass > 0.0);
    }

    #[test]
    fn test_ball_joint_separation() {
        let mut j = BallJoint::new(0, 1, Vec3::X, Vec3::NEG_X);
        j.prepare(
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY,
            1.0, 1.0
        );
        // Anchor A at (1,0,0), Anchor B at (1,0,0) -> separation = 0
        assert!(j.separation().length() < 0.01);
    }

    #[test]
    fn test_ball_joint_solve_position() {
        let mut j = BallJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO);
        let result = j.solve_position(
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY,
            1.0, 1.0
        );
        
        // Bodies should move toward each other
        assert!(result.delta_pos_a.x > 0.0);
        assert!(result.delta_pos_b.x < 0.0);
    }

    #[test]
    fn test_ball_joint_no_correction_when_aligned() {
        let mut j = BallJoint::new(0, 1, Vec3::X, Vec3::NEG_X);
        let result = j.solve_position(
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY,
            1.0, 1.0
        );
        
        // Anchors are at same position, no correction needed
        assert!(result.delta_pos_a.length() < 0.01);
    }

    #[test]
    fn test_ball_joint_warm_start_empty() {
        let j = BallJoint::new(0, 1, Vec3::X, Vec3::NEG_X);
        let impulses = j.warm_start();
        assert!(impulses.is_empty());
    }
}
