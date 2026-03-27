// Physics step - main simulation loop

use glam::Vec3;

/// Configuration for physics step
#[derive(Debug, Clone)]
pub struct StepConfig {
    /// Fixed timestep in seconds
    pub dt: f32,
    /// Maximum substeps per frame
    pub max_substeps: usize,
    /// Gravity vector
    pub gravity: Vec3,
    /// Velocity solver iterations
    pub velocity_iterations: usize,
    /// Position solver iterations
    pub position_iterations: usize,
    /// Enable sleeping
    pub allow_sleeping: bool,
    /// Sleep velocity threshold
    pub sleep_threshold: f32,
    /// Time to sleep after being still
    pub time_to_sleep: f32,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            max_substeps: 8,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            velocity_iterations: 8,
            position_iterations: 3,
            allow_sleeping: true,
            sleep_threshold: 0.05,
            time_to_sleep: 0.5,
        }
    }
}

impl StepConfig {
    pub fn new(dt: f32) -> Self {
        Self {
            dt,
            ..Default::default()
        }
    }
    
    pub fn with_gravity(mut self, gravity: Vec3) -> Self {
        self.gravity = gravity;
        self
    }
    
    pub fn with_iterations(mut self, velocity: usize, position: usize) -> Self {
        self.velocity_iterations = velocity;
        self.position_iterations = position;
        self
    }
}

/// Step timing info
#[derive(Debug, Clone, Copy, Default)]
pub struct StepTiming {
    pub broad_phase_ms: f32,
    pub narrow_phase_ms: f32,
    pub solver_ms: f32,
    pub integration_ms: f32,
    pub total_ms: f32,
}

/// Results of a physics step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Number of substeps taken
    pub substeps: usize,
    /// Number of contact pairs
    pub contact_count: usize,
    /// Number of islands
    pub island_count: usize,
    /// Timing information
    pub timing: StepTiming,
    /// Bodies that went to sleep
    pub newly_sleeping: Vec<u32>,
    /// Bodies that woke up
    pub newly_awake: Vec<u32>,
}

impl StepResult {
    pub fn new() -> Self {
        Self {
            substeps: 0,
            contact_count: 0,
            island_count: 0,
            timing: StepTiming::default(),
            newly_sleeping: Vec::new(),
            newly_awake: Vec::new(),
        }
    }
}

impl Default for StepResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Main physics step function signature
/// This is a trait for the physics world to implement
pub trait PhysicsStep {
    /// Perform a single physics step
    fn step(&mut self, config: &StepConfig) -> StepResult;
    
    /// Perform substeps for variable frame time
    fn step_variable(&mut self, config: &StepConfig, frame_time: f32) -> StepResult {
        let mut accumulated = frame_time;
        let mut result = StepResult::new();
        
        while accumulated >= config.dt && result.substeps < config.max_substeps {
            let substep_result = self.step(config);
            
            result.substeps += 1;
            result.contact_count = substep_result.contact_count;
            result.island_count = substep_result.island_count;
            result.timing.total_ms += substep_result.timing.total_ms;
            
            accumulated -= config.dt;
        }
        
        result
    }
}

/// Velocity integration
pub fn integrate_velocities(
    velocities: &mut [Vec3],
    angular_velocities: &mut [Vec3],
    forces: &[Vec3],
    torques: &[Vec3],
    inv_masses: &[f32],
    inv_inertias: &[Vec3],
    gravity: Vec3,
    dt: f32,
) {
    for i in 0..velocities.len() {
        if inv_masses[i] > 0.0 {
            // Apply gravity
            velocities[i] += gravity * dt;
            
            // Apply forces
            velocities[i] += forces[i] * inv_masses[i] * dt;
            
            // Apply torques
            angular_velocities[i] += torques[i] * inv_inertias[i] * dt;
        }
    }
}

/// Position integration
pub fn integrate_positions(
    positions: &mut [Vec3],
    orientations: &mut [glam::Quat],
    velocities: &[Vec3],
    angular_velocities: &[Vec3],
    dt: f32,
) {
    for i in 0..positions.len() {
        positions[i] += velocities[i] * dt;
        
        // Integrate orientation using quaternion derivative
        let omega_quat = glam::Quat::from_xyzw(
            angular_velocities[i].x * 0.5,
            angular_velocities[i].y * 0.5,
            angular_velocities[i].z * 0.5,
            0.0,
        );
        let delta = omega_quat * orientations[i] * dt;
        orientations[i] = glam::Quat::from_xyzw(
            orientations[i].x + delta.x,
            orientations[i].y + delta.y,
            orientations[i].z + delta.z,
            orientations[i].w + delta.w,
        ).normalize();
    }
}

/// Apply damping
pub fn apply_damping(
    velocities: &mut [Vec3],
    angular_velocities: &mut [Vec3],
    linear_damping: f32,
    angular_damping: f32,
    dt: f32,
) {
    let linear_factor = (1.0 - linear_damping).max(0.0).powf(dt);
    let angular_factor = (1.0 - angular_damping).max(0.0).powf(dt);
    
    for v in velocities.iter_mut() {
        *v *= linear_factor;
    }
    
    for av in angular_velocities.iter_mut() {
        *av *= angular_factor;
    }
}

/// Check if body should sleep
pub fn should_sleep(
    velocity: Vec3,
    angular_velocity: Vec3,
    sleep_time: f32,
    threshold: f32,
    time_to_sleep: f32,
) -> bool {
    let linear_energy = velocity.length_squared();
    let angular_energy = angular_velocity.length_squared();
    let total_energy = linear_energy + angular_energy;
    
    total_energy < threshold * threshold && sleep_time >= time_to_sleep
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn test_step_config_default() {
        let config = StepConfig::default();
        assert!((config.dt - 1.0 / 60.0).abs() < 0.001);
        assert_eq!(config.velocity_iterations, 8);
    }

    #[test]
    fn test_step_config_with_gravity() {
        let config = StepConfig::default().with_gravity(Vec3::new(0.0, -20.0, 0.0));
        assert_eq!(config.gravity.y, -20.0);
    }

    #[test]
    fn test_step_result_new() {
        let result = StepResult::new();
        assert_eq!(result.substeps, 0);
        assert_eq!(result.contact_count, 0);
    }

    #[test]
    fn test_integrate_velocities() {
        let mut velocities = vec![Vec3::ZERO];
        let mut angular_velocities = vec![Vec3::ZERO];
        let forces = vec![Vec3::ZERO];
        let torques = vec![Vec3::ZERO];
        let inv_masses = vec![1.0];
        let inv_inertias = vec![Vec3::ONE];
        
        integrate_velocities(
            &mut velocities,
            &mut angular_velocities,
            &forces,
            &torques,
            &inv_masses,
            &inv_inertias,
            Vec3::new(0.0, -10.0, 0.0),
            1.0,
        );
        
        assert_eq!(velocities[0], Vec3::new(0.0, -10.0, 0.0));
    }

    #[test]
    fn test_integrate_positions() {
        let mut positions = vec![Vec3::ZERO];
        let mut orientations = vec![Quat::IDENTITY];
        let velocities = vec![Vec3::new(10.0, 0.0, 0.0)];
        let angular_velocities = vec![Vec3::ZERO];
        
        integrate_positions(
            &mut positions,
            &mut orientations,
            &velocities,
            &angular_velocities,
            1.0,
        );
        
        assert_eq!(positions[0], Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn test_apply_damping() {
        let mut velocities = vec![Vec3::new(10.0, 0.0, 0.0)];
        let mut angular_velocities = vec![Vec3::new(0.0, 10.0, 0.0)];
        
        apply_damping(&mut velocities, &mut angular_velocities, 0.1, 0.1, 1.0);
        
        assert!(velocities[0].x < 10.0);
        assert!(angular_velocities[0].y < 10.0);
    }

    #[test]
    fn test_should_sleep_yes() {
        assert!(should_sleep(
            Vec3::new(0.01, 0.0, 0.0),
            Vec3::ZERO,
            1.0,
            0.05,
            0.5,
        ));
    }

    #[test]
    fn test_should_sleep_no_moving() {
        assert!(!should_sleep(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            1.0,
            0.05,
            0.5,
        ));
    }

    #[test]
    fn test_should_sleep_no_time() {
        assert!(!should_sleep(
            Vec3::new(0.01, 0.0, 0.0),
            Vec3::ZERO,
            0.1,
            0.05,
            0.5,
        ));
    }

    #[test]
    fn test_static_body_not_integrated() {
        let mut velocities = vec![Vec3::ZERO];
        let mut angular_velocities = vec![Vec3::ZERO];
        let forces = vec![Vec3::new(100.0, 0.0, 0.0)];
        let torques = vec![Vec3::ZERO];
        let inv_masses = vec![0.0]; // Static
        let inv_inertias = vec![Vec3::ZERO];
        
        integrate_velocities(
            &mut velocities,
            &mut angular_velocities,
            &forces,
            &torques,
            &inv_masses,
            &inv_inertias,
            Vec3::new(0.0, -10.0, 0.0),
            1.0,
        );
        
        assert_eq!(velocities[0], Vec3::ZERO);
    }
}
