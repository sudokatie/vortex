//! 2D rigid body

use glam::Vec2;
use super::{Shape2D, Transform2D};

/// Body type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType2D {
    Dynamic,
    Static,
    Kinematic,
}

/// 2D rigid body
#[derive(Debug, Clone)]
pub struct RigidBody2D {
    // Transform
    pub position: Vec2,
    pub rotation: f32,
    
    // Velocity
    pub velocity: Vec2,
    pub angular_velocity: f32,
    
    // Mass properties
    pub mass: f32,
    pub inv_mass: f32,
    pub inertia: f32,
    pub inv_inertia: f32,
    
    // Accumulated forces
    pub force: Vec2,
    pub torque: f32,
    
    // Shape
    pub shape: Shape2D,
    pub body_type: BodyType2D,
    
    // State
    pub is_sleeping: bool,
    pub sleep_time: f32,
    
    // Material
    pub friction: f32,
    pub restitution: f32,
}

impl RigidBody2D {
    /// Create a new rigid body
    pub fn new(shape: Shape2D, mass: f32, body_type: BodyType2D) -> Self {
        let (inv_mass, inertia, inv_inertia) = match body_type {
            BodyType2D::Dynamic => {
                let inv_m = if mass > 0.0 { 1.0 / mass } else { 0.0 };
                let i = shape.moment_of_inertia(mass);
                let inv_i = if i > 0.0 { 1.0 / i } else { 0.0 };
                (inv_m, i, inv_i)
            }
            BodyType2D::Static | BodyType2D::Kinematic => {
                (0.0, 0.0, 0.0)
            }
        };
        
        Self {
            position: Vec2::ZERO,
            rotation: 0.0,
            velocity: Vec2::ZERO,
            angular_velocity: 0.0,
            mass,
            inv_mass,
            inertia,
            inv_inertia,
            force: Vec2::ZERO,
            torque: 0.0,
            shape,
            body_type,
            is_sleeping: false,
            sleep_time: 0.0,
            friction: 0.5,
            restitution: 0.3,
        }
    }
    
    /// Create a dynamic body
    pub fn dynamic(shape: Shape2D, mass: f32) -> Self {
        Self::new(shape, mass, BodyType2D::Dynamic)
    }
    
    /// Create a static body
    pub fn stationary(shape: Shape2D) -> Self {
        Self::new(shape, 0.0, BodyType2D::Static)
    }
    
    /// Create a kinematic body
    pub fn kinematic(shape: Shape2D) -> Self {
        Self::new(shape, 0.0, BodyType2D::Kinematic)
    }
    
    /// Get the current transform
    pub fn transform(&self) -> Transform2D {
        Transform2D::new(self.position, self.rotation)
    }
    
    /// Apply force at center of mass
    pub fn apply_force(&mut self, force: Vec2) {
        if self.body_type == BodyType2D::Dynamic {
            self.force += force;
            self.wake();
        }
    }
    
    /// Apply torque
    pub fn apply_torque(&mut self, torque: f32) {
        if self.body_type == BodyType2D::Dynamic {
            self.torque += torque;
            self.wake();
        }
    }
    
    /// Apply force at world point
    pub fn apply_force_at_point(&mut self, force: Vec2, point: Vec2) {
        if self.body_type == BodyType2D::Dynamic {
            self.force += force;
            let r = point - self.position;
            self.torque += r.perp_dot(force);
            self.wake();
        }
    }
    
    /// Apply impulse at center of mass
    pub fn apply_impulse(&mut self, impulse: Vec2) {
        if self.body_type == BodyType2D::Dynamic {
            self.velocity += impulse * self.inv_mass;
            self.wake();
        }
    }
    
    /// Apply impulse at world point
    pub fn apply_impulse_at_point(&mut self, impulse: Vec2, point: Vec2) {
        if self.body_type == BodyType2D::Dynamic {
            self.velocity += impulse * self.inv_mass;
            let r = point - self.position;
            self.angular_velocity += r.perp_dot(impulse) * self.inv_inertia;
            self.wake();
        }
    }
    
    /// Apply angular impulse
    pub fn apply_angular_impulse(&mut self, impulse: f32) {
        if self.body_type == BodyType2D::Dynamic {
            self.angular_velocity += impulse * self.inv_inertia;
            self.wake();
        }
    }
    
    /// Clear accumulated forces
    pub fn clear_forces(&mut self) {
        self.force = Vec2::ZERO;
        self.torque = 0.0;
    }
    
    /// Get velocity at world point
    pub fn velocity_at_point(&self, point: Vec2) -> Vec2 {
        let r = point - self.position;
        self.velocity + Vec2::new(-r.y, r.x) * self.angular_velocity
    }
    
    /// Integrate forces to velocity
    pub fn integrate_velocity(&mut self, dt: f32, gravity: Vec2) {
        if self.body_type != BodyType2D::Dynamic || self.is_sleeping {
            return;
        }
        
        // Apply gravity
        let acceleration = gravity + self.force * self.inv_mass;
        self.velocity += acceleration * dt;
        
        // Apply torque
        self.angular_velocity += self.torque * self.inv_inertia * dt;
        
        self.clear_forces();
    }
    
    /// Integrate velocity to position
    pub fn integrate_position(&mut self, dt: f32) {
        if self.body_type == BodyType2D::Static || self.is_sleeping {
            return;
        }
        
        self.position += self.velocity * dt;
        self.rotation += self.angular_velocity * dt;
        
        // Normalize rotation to [-PI, PI]
        while self.rotation > std::f32::consts::PI {
            self.rotation -= std::f32::consts::TAU;
        }
        while self.rotation < -std::f32::consts::PI {
            self.rotation += std::f32::consts::TAU;
        }
    }
    
    /// Wake up from sleeping
    pub fn wake(&mut self) {
        if self.is_sleeping {
            self.is_sleeping = false;
            self.sleep_time = 0.0;
        }
    }
    
    /// Put to sleep
    pub fn sleep(&mut self) {
        self.is_sleeping = true;
        self.velocity = Vec2::ZERO;
        self.angular_velocity = 0.0;
    }
    
    /// Update sleep state
    pub fn update_sleep(&mut self, dt: f32, threshold: f32, time_to_sleep: f32) {
        if self.body_type != BodyType2D::Dynamic {
            return;
        }
        
        let energy = self.velocity.length_squared() + self.angular_velocity * self.angular_velocity;
        
        if energy < threshold {
            self.sleep_time += dt;
            if self.sleep_time > time_to_sleep {
                self.sleep();
            }
        } else {
            self.sleep_time = 0.0;
        }
    }
    
    pub fn is_dynamic(&self) -> bool {
        self.body_type == BodyType2D::Dynamic
    }
    
    pub fn is_static(&self) -> bool {
        self.body_type == BodyType2D::Static
    }
    
    pub fn is_kinematic(&self) -> bool {
        self.body_type == BodyType2D::Kinematic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_body_dynamic() {
        let body = RigidBody2D::dynamic(Shape2D::circle(1.0), 1.0);
        assert!(body.is_dynamic());
        assert!(body.inv_mass > 0.0);
    }
    
    #[test]
    fn test_body_static() {
        let body = RigidBody2D::stationary(Shape2D::circle(1.0));
        assert!(body.is_static());
        assert_eq!(body.inv_mass, 0.0);
    }
    
    #[test]
    fn test_apply_force() {
        let mut body = RigidBody2D::dynamic(Shape2D::circle(1.0), 1.0);
        body.apply_force(Vec2::new(10.0, 0.0));
        assert_eq!(body.force, Vec2::new(10.0, 0.0));
    }
    
    #[test]
    fn test_apply_impulse() {
        let mut body = RigidBody2D::dynamic(Shape2D::circle(1.0), 1.0);
        body.apply_impulse(Vec2::new(10.0, 0.0));
        assert!(body.velocity.x > 0.0);
    }
    
    #[test]
    fn test_integrate_velocity() {
        let mut body = RigidBody2D::dynamic(Shape2D::circle(1.0), 1.0);
        body.integrate_velocity(1.0, Vec2::new(0.0, -10.0));
        assert_eq!(body.velocity, Vec2::new(0.0, -10.0));
    }
    
    #[test]
    fn test_integrate_position() {
        let mut body = RigidBody2D::dynamic(Shape2D::circle(1.0), 1.0);
        body.velocity = Vec2::new(10.0, 0.0);
        body.integrate_position(1.0);
        assert_eq!(body.position, Vec2::new(10.0, 0.0));
    }
    
    #[test]
    fn test_velocity_at_point() {
        let mut body = RigidBody2D::dynamic(Shape2D::circle(1.0), 1.0);
        body.velocity = Vec2::new(1.0, 0.0);
        body.angular_velocity = 1.0;
        
        let v = body.velocity_at_point(Vec2::new(0.0, 1.0));
        assert!(v.x > 1.0); // Linear + angular contribution
    }
    
    #[test]
    fn test_sleep() {
        let mut body = RigidBody2D::dynamic(Shape2D::circle(1.0), 1.0);
        body.velocity = Vec2::ONE;
        body.sleep();
        
        assert!(body.is_sleeping);
        assert_eq!(body.velocity, Vec2::ZERO);
    }
}
