// Numerical integrators for physics simulation

use glam::{Quat, Vec3};

/// Integration method
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IntegratorType {
    /// Semi-implicit Euler (symplectic, good energy conservation)
    #[default]
    SemiImplicitEuler,
    /// Explicit Euler (simplest, less stable)
    ExplicitEuler,
    /// Velocity Verlet (second-order accurate)
    Verlet,
}

/// Body state for integration
#[derive(Debug, Clone, Copy)]
pub struct IntegrationState {
    pub position: Vec3,
    pub velocity: Vec3,
    pub orientation: Quat,
    pub angular_velocity: Vec3,
}

impl IntegrationState {
    pub fn new(position: Vec3, velocity: Vec3) -> Self {
        Self {
            position,
            velocity,
            orientation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        }
    }
    
    pub fn with_rotation(mut self, orientation: Quat, angular_velocity: Vec3) -> Self {
        self.orientation = orientation;
        self.angular_velocity = angular_velocity;
        self
    }
}

/// Forces/accelerations to apply
#[derive(Debug, Clone, Copy, Default)]
pub struct IntegrationInput {
    pub linear_acceleration: Vec3,
    pub angular_acceleration: Vec3,
}

impl IntegrationInput {
    pub fn linear(acceleration: Vec3) -> Self {
        Self {
            linear_acceleration: acceleration,
            angular_acceleration: Vec3::ZERO,
        }
    }
    
    pub fn full(linear: Vec3, angular: Vec3) -> Self {
        Self {
            linear_acceleration: linear,
            angular_acceleration: angular,
        }
    }
}

/// Integrate velocity given acceleration
pub fn integrate_velocity(
    integrator: IntegratorType,
    velocity: Vec3,
    acceleration: Vec3,
    dt: f32,
) -> Vec3 {
    match integrator {
        IntegratorType::ExplicitEuler |
        IntegratorType::SemiImplicitEuler |
        IntegratorType::Verlet => {
            velocity + acceleration * dt
        }
    }
}

/// Integrate position given velocity
pub fn integrate_position(
    integrator: IntegratorType,
    position: Vec3,
    velocity: Vec3,
    velocity_new: Vec3,
    acceleration: Vec3,
    dt: f32,
) -> Vec3 {
    match integrator {
        IntegratorType::ExplicitEuler => {
            // Use old velocity
            position + velocity * dt
        }
        IntegratorType::SemiImplicitEuler => {
            // Use new velocity (more stable)
            position + velocity_new * dt
        }
        IntegratorType::Verlet => {
            // Second-order accurate
            position + velocity * dt + 0.5 * acceleration * dt * dt
        }
    }
}

/// Integrate angular velocity
pub fn integrate_angular_velocity(
    angular_velocity: Vec3,
    angular_acceleration: Vec3,
    dt: f32,
) -> Vec3 {
    angular_velocity + angular_acceleration * dt
}

/// Integrate orientation quaternion
pub fn integrate_orientation(
    orientation: Quat,
    angular_velocity: Vec3,
    dt: f32,
) -> Quat {
    let omega_quat = Quat::from_xyzw(
        angular_velocity.x,
        angular_velocity.y,
        angular_velocity.z,
        0.0,
    );
    
    let delta = omega_quat * orientation * 0.5 * dt;
    let new_orientation = Quat::from_xyzw(
        orientation.x + delta.x,
        orientation.y + delta.y,
        orientation.z + delta.z,
        orientation.w + delta.w,
    );
    
    new_orientation.normalize()
}

/// Full state integration step
pub fn integrate_state(
    integrator: IntegratorType,
    state: IntegrationState,
    input: IntegrationInput,
    dt: f32,
) -> IntegrationState {
    // Integrate velocities
    let new_velocity = integrate_velocity(
        integrator,
        state.velocity,
        input.linear_acceleration,
        dt,
    );
    let new_angular_velocity = integrate_angular_velocity(
        state.angular_velocity,
        input.angular_acceleration,
        dt,
    );
    
    // Integrate positions
    let new_position = integrate_position(
        integrator,
        state.position,
        state.velocity,
        new_velocity,
        input.linear_acceleration,
        dt,
    );
    let new_orientation = integrate_orientation(
        state.orientation,
        new_angular_velocity,
        dt,
    );
    
    IntegrationState {
        position: new_position,
        velocity: new_velocity,
        orientation: new_orientation,
        angular_velocity: new_angular_velocity,
    }
}

/// Damping helpers
pub fn apply_linear_damping(velocity: Vec3, damping: f32, dt: f32) -> Vec3 {
    let factor = (1.0 - damping).powf(dt);
    velocity * factor
}

pub fn apply_angular_damping(angular_velocity: Vec3, damping: f32, dt: f32) -> Vec3 {
    let factor = (1.0 - damping).powf(dt);
    angular_velocity * factor
}

/// Clamp velocity magnitudes
pub fn clamp_velocity(velocity: Vec3, max_speed: f32) -> Vec3 {
    let speed = velocity.length();
    if speed > max_speed {
        velocity * (max_speed / speed)
    } else {
        velocity
    }
}

pub fn clamp_angular_velocity(angular_velocity: Vec3, max_angular_speed: f32) -> Vec3 {
    let speed = angular_velocity.length();
    if speed > max_angular_speed {
        angular_velocity * (max_angular_speed / speed)
    } else {
        angular_velocity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_velocity_euler() {
        let v = integrate_velocity(
            IntegratorType::ExplicitEuler,
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            1.0,
        );
        assert_eq!(v, Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn test_integrate_position_explicit() {
        let p = integrate_position(
            IntegratorType::ExplicitEuler,
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(20.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            1.0,
        );
        // Uses old velocity
        assert_eq!(p, Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn test_integrate_position_semi_implicit() {
        let p = integrate_position(
            IntegratorType::SemiImplicitEuler,
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(20.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            1.0,
        );
        // Uses new velocity
        assert_eq!(p, Vec3::new(20.0, 0.0, 0.0));
    }

    #[test]
    fn test_integrate_position_verlet() {
        let p = integrate_position(
            IntegratorType::Verlet,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            1.0,
        );
        // p + v*dt + 0.5*a*dt^2 = 0 + 0 + 5 = 5
        assert_eq!(p, Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn test_integrate_orientation() {
        let q = integrate_orientation(Quat::IDENTITY, Vec3::new(0.0, 1.0, 0.0), 0.1);
        assert!(q.is_normalized());
        assert!(q.w < 1.0); // Changed from identity
    }

    #[test]
    fn test_integrate_state() {
        let state = IntegrationState::new(Vec3::ZERO, Vec3::ZERO);
        let input = IntegrationInput::linear(Vec3::new(0.0, -10.0, 0.0));
        let new_state = integrate_state(IntegratorType::SemiImplicitEuler, state, input, 1.0);
        
        assert_eq!(new_state.velocity, Vec3::new(0.0, -10.0, 0.0));
        assert_eq!(new_state.position, Vec3::new(0.0, -10.0, 0.0));
    }

    #[test]
    fn test_linear_damping() {
        let v = apply_linear_damping(Vec3::new(100.0, 0.0, 0.0), 0.1, 1.0);
        assert!(v.x < 100.0);
        assert!(v.x > 80.0);
    }

    #[test]
    fn test_clamp_velocity() {
        let v = clamp_velocity(Vec3::new(100.0, 0.0, 0.0), 10.0);
        assert!((v.length() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_clamp_velocity_under() {
        let v = clamp_velocity(Vec3::new(5.0, 0.0, 0.0), 10.0);
        assert_eq!(v, Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn test_angular_damping() {
        let v = apply_angular_damping(Vec3::new(10.0, 0.0, 0.0), 0.5, 1.0);
        assert!(v.x < 10.0);
    }
}
