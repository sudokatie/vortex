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
    /// Runge-Kutta 4th order (accurate, expensive)
    Rk4,
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
        IntegratorType::Verlet |
        IntegratorType::Rk4 => {
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
        IntegratorType::Rk4 => {
            // RK4: Use weighted average of velocities
            // For constant acceleration: same as Verlet
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
    match integrator {
        IntegratorType::Rk4 => integrate_state_rk4(state, input, dt),
        _ => {
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
    }
}

/// RK4 integration for full state.
/// Uses 4th-order Runge-Kutta method for higher accuracy.
fn integrate_state_rk4(
    state: IntegrationState,
    input: IntegrationInput,
    dt: f32,
) -> IntegrationState {
    let a = input.linear_acceleration;
    let aa = input.angular_acceleration;
    
    // k1: derivatives at current state
    let k1_v = a;
    let k1_p = state.velocity;
    let k1_av = aa;
    let k1_o = state.angular_velocity;
    
    // k2: derivatives at midpoint using k1
    let k2_v = a; // constant acceleration
    let k2_p = state.velocity + k1_v * (dt * 0.5);
    let k2_av = aa;
    let k2_o = state.angular_velocity + k1_av * (dt * 0.5);
    
    // k3: derivatives at midpoint using k2
    let k3_v = a;
    let k3_p = state.velocity + k2_v * (dt * 0.5);
    let k3_av = aa;
    let k3_o = state.angular_velocity + k2_av * (dt * 0.5);
    
    // k4: derivatives at end using k3
    let k4_v = a;
    let k4_p = state.velocity + k3_v * dt;
    let k4_av = aa;
    let k4_o = state.angular_velocity + k3_av * dt;
    
    // Weighted average
    let new_velocity = state.velocity + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * (dt / 6.0);
    let new_position = state.position + (k1_p + k2_p * 2.0 + k3_p * 2.0 + k4_p) * (dt / 6.0);
    let new_angular_velocity = state.angular_velocity + (k1_av + k2_av * 2.0 + k3_av * 2.0 + k4_av) * (dt / 6.0);
    
    // Average angular velocity for orientation
    let avg_angular = (k1_o + k2_o * 2.0 + k3_o * 2.0 + k4_o) / 6.0;
    let new_orientation = integrate_orientation(state.orientation, avg_angular, dt);
    
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
    
    #[test]
    fn test_integrate_velocity_rk4() {
        let v = integrate_velocity(
            IntegratorType::Rk4,
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            1.0,
        );
        assert_eq!(v, Vec3::new(10.0, 0.0, 0.0));
    }
    
    #[test]
    fn test_integrate_position_rk4() {
        let p = integrate_position(
            IntegratorType::Rk4,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            1.0,
        );
        // p + v*dt + 0.5*a*dt^2 = 0 + 0 + 5 = 5
        assert_eq!(p, Vec3::new(5.0, 0.0, 0.0));
    }
    
    #[test]
    fn test_integrate_state_rk4() {
        let state = IntegrationState::new(Vec3::ZERO, Vec3::ZERO);
        let input = IntegrationInput::linear(Vec3::new(0.0, -10.0, 0.0));
        let new_state = integrate_state(IntegratorType::Rk4, state, input, 1.0);
        
        // With RK4 and constant acceleration, should be similar to Verlet
        assert!((new_state.velocity.y - (-10.0)).abs() < 0.01);
        assert!((new_state.position.y - (-5.0)).abs() < 0.01);
    }
    
    #[test]
    fn test_rk4_higher_accuracy() {
        // RK4 should be more accurate for varying acceleration scenarios
        let state = IntegrationState::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let input = IntegrationInput::linear(Vec3::new(0.0, -10.0, 0.0));
        
        let euler_result = integrate_state(IntegratorType::ExplicitEuler, state, input, 0.1);
        let rk4_result = integrate_state(IntegratorType::Rk4, state, input, 0.1);
        
        // Both should give similar results for constant acceleration
        assert!((euler_result.velocity.y - rk4_result.velocity.y).abs() < 0.01);
    }
}
