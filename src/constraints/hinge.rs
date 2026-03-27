// Hinge joint (revolute joint) - rotation around a single axis

use glam::{Quat, Vec3};
use super::joint::{Joint, JointImpulse, JointLimit, JointMotor, PositionResult};

/// Hinge joint - allows rotation around a single axis
#[derive(Debug, Clone)]
pub struct HingeJoint {
    body_a: u32,
    body_b: u32,
    /// Local anchor on body A
    local_anchor_a: Vec3,
    /// Local anchor on body B
    local_anchor_b: Vec3,
    /// Local axis on body A
    local_axis_a: Vec3,
    /// Local axis on body B
    local_axis_b: Vec3,
    /// Angle limits
    limit: JointLimit,
    /// Motor
    motor: JointMotor,
    /// Accumulated linear impulse
    linear_impulse: Vec3,
    /// Accumulated angular impulse (for axis alignment)
    angular_impulse: Vec3,
    /// Accumulated limit impulse
    limit_impulse: f32,
    /// Accumulated motor impulse
    motor_impulse: f32,
    /// Current angle
    current_angle: f32,
    /// Effective mass
    mass: f32,
}

impl HingeJoint {
    pub fn new(
        body_a: u32,
        body_b: u32,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        local_axis_a: Vec3,
        local_axis_b: Vec3,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a,
            local_anchor_b,
            local_axis_a: local_axis_a.normalize(),
            local_axis_b: local_axis_b.normalize(),
            limit: JointLimit::disabled(),
            motor: JointMotor::disabled(),
            linear_impulse: Vec3::ZERO,
            angular_impulse: Vec3::ZERO,
            limit_impulse: 0.0,
            motor_impulse: 0.0,
            current_angle: 0.0,
            mass: 0.0,
        }
    }
    
    pub fn with_limit(mut self, lower: f32, upper: f32) -> Self {
        self.limit = JointLimit::new(lower, upper);
        self
    }
    
    pub fn with_motor(mut self, target_velocity: f32, max_force: f32) -> Self {
        self.motor = JointMotor::new(target_velocity, max_force);
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
        let world_axis_a = rot_a * self.local_axis_a;
        let world_axis_b = rot_b * self.local_axis_b;
        
        // Compute current angle
        self.current_angle = self.compute_angle(rot_a, rot_b);
        
        // Simplified effective mass
        let total = inv_mass_a + inv_mass_b;
        self.mass = if total > 0.0 { 1.0 / total } else { 0.0 };
        
        let _ = (pos_a, pos_b, world_axis_a, world_axis_b);
    }
    
    /// Compute current joint angle
    pub fn compute_angle(&self, rot_a: Quat, rot_b: Quat) -> f32 {
        let world_axis_a = rot_a * self.local_axis_a;
        
        // Reference vectors perpendicular to axis
        let ref_a = if world_axis_a.x.abs() < 0.9 {
            world_axis_a.cross(Vec3::X).normalize()
        } else {
            world_axis_a.cross(Vec3::Y).normalize()
        };
        
        let ref_b = rot_b * (rot_a.inverse() * ref_a);
        
        // Angle between reference vectors
        let cos_angle = ref_a.dot(ref_b).clamp(-1.0, 1.0);
        let sin_angle = world_axis_a.dot(ref_a.cross(ref_b));
        
        sin_angle.atan2(cos_angle)
    }
    
    /// Get current angle
    pub fn angle(&self) -> f32 {
        self.current_angle
    }
    
    /// Get angular velocity along hinge axis
    pub fn angular_velocity(&self, ang_vel_a: Vec3, ang_vel_b: Vec3, rot_a: Quat) -> f32 {
        let world_axis = rot_a * self.local_axis_a;
        (ang_vel_b - ang_vel_a).dot(world_axis)
    }
}

impl Joint for HingeJoint {
    fn bodies(&self) -> (u32, u32) {
        (self.body_a, self.body_b)
    }
    
    fn warm_start(&self) -> Vec<JointImpulse> {
        let mut impulses = Vec::new();
        
        if self.linear_impulse.length_squared() > 1e-12 {
            impulses.push(JointImpulse::new(
                self.body_a,
                self.body_b,
                self.linear_impulse,
            ));
        }
        
        impulses
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
        let mut impulses = Vec::new();
        
        // Point-to-point constraint (same as ball joint)
        let rel_vel = vel_b - vel_a;
        let lambda = -self.mass * rel_vel;
        self.linear_impulse += lambda;
        
        impulses.push(JointImpulse::new(self.body_a, self.body_b, lambda));
        
        // Motor constraint
        if self.motor.enabled {
            // Simplified: just apply angular impulse along axis
            // In a full implementation, we'd use the proper effective mass
            let _ = (ang_vel_a, ang_vel_b);
        }
        
        impulses
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
        // Position constraint (anchor points must coincide)
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
        
        let correction = error * 0.2;
        let delta_a = correction * (inv_mass_a / total_inv_mass);
        let delta_b = -correction * (inv_mass_b / total_inv_mass);
        
        PositionResult::position_only(delta_a, delta_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_hinge_joint_new() {
        let j = HingeJoint::new(
            0, 1,
            Vec3::ZERO, Vec3::ZERO,
            Vec3::Y, Vec3::Y
        );
        assert_eq!(j.bodies(), (0, 1));
    }

    #[test]
    fn test_hinge_joint_with_limit() {
        let j = HingeJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y)
            .with_limit(-PI / 4.0, PI / 4.0);
        assert!(j.limit.enabled);
    }

    #[test]
    fn test_hinge_joint_with_motor() {
        let j = HingeJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y)
            .with_motor(10.0, 100.0);
        assert!(j.motor.enabled);
    }

    #[test]
    fn test_hinge_joint_angle_zero() {
        let j = HingeJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y);
        let angle = j.compute_angle(Quat::IDENTITY, Quat::IDENTITY);
        assert!(angle.abs() < 0.1);
    }

    #[test]
    fn test_hinge_joint_angle_rotated() {
        let j = HingeJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y);
        let rot_b = Quat::from_rotation_y(PI / 2.0);
        let angle = j.compute_angle(Quat::IDENTITY, rot_b);
        assert!((angle - PI / 2.0).abs() < 0.1);
    }

    #[test]
    fn test_hinge_joint_prepare() {
        let mut j = HingeJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y);
        j.prepare(
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::X, Quat::IDENTITY,
            1.0, 1.0
        );
        assert!(j.mass > 0.0);
    }

    #[test]
    fn test_hinge_joint_angular_velocity() {
        let j = HingeJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y);
        let ang_vel = j.angular_velocity(
            Vec3::ZERO,
            Vec3::new(0.0, 5.0, 0.0),
            Quat::IDENTITY
        );
        assert!((ang_vel - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_hinge_joint_solve_position() {
        let mut j = HingeJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y);
        let result = j.solve_position(
            Vec3::ZERO, Quat::IDENTITY,
            Vec3::X, Quat::IDENTITY,
            1.0, 1.0
        );
        
        assert!(result.delta_pos_a.x > 0.0);
        assert!(result.delta_pos_b.x < 0.0);
    }
}
