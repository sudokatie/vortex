// Force generators for physics simulation

use glam::Vec3;

/// Force generator trait - applies forces to bodies
pub trait ForceGenerator: Send + Sync {
    /// Apply force to a body given its state
    fn apply(&self, position: Vec3, velocity: Vec3, mass: f32) -> ForceOutput;
    
    /// Whether this generator is still active
    fn is_active(&self) -> bool {
        true
    }
}

/// Output from force generator
#[derive(Debug, Clone, Copy, Default)]
pub struct ForceOutput {
    pub force: Vec3,
    pub torque: Vec3,
}

impl ForceOutput {
    pub fn force(force: Vec3) -> Self {
        Self { force, torque: Vec3::ZERO }
    }
    
    pub fn with_torque(force: Vec3, torque: Vec3) -> Self {
        Self { force, torque }
    }
}

/// Constant gravity force
#[derive(Debug, Clone, Copy)]
pub struct Gravity {
    pub acceleration: Vec3,
}

impl Gravity {
    pub fn new(acceleration: Vec3) -> Self {
        Self { acceleration }
    }
    
    pub fn earth() -> Self {
        Self::new(Vec3::new(0.0, -9.81, 0.0))
    }
    
    pub fn moon() -> Self {
        Self::new(Vec3::new(0.0, -1.62, 0.0))
    }
}

impl ForceGenerator for Gravity {
    fn apply(&self, _position: Vec3, _velocity: Vec3, mass: f32) -> ForceOutput {
        ForceOutput::force(self.acceleration * mass)
    }
}

/// Spring force between anchor and body
#[derive(Debug, Clone, Copy)]
pub struct Spring {
    pub anchor: Vec3,
    pub rest_length: f32,
    pub stiffness: f32,
    pub damping: f32,
}

impl Spring {
    pub fn new(anchor: Vec3, rest_length: f32, stiffness: f32, damping: f32) -> Self {
        Self { anchor, rest_length, stiffness, damping }
    }
    
    pub fn soft(anchor: Vec3, rest_length: f32) -> Self {
        Self::new(anchor, rest_length, 10.0, 1.0)
    }
    
    pub fn stiff(anchor: Vec3, rest_length: f32) -> Self {
        Self::new(anchor, rest_length, 500.0, 10.0)
    }
}

impl ForceGenerator for Spring {
    fn apply(&self, position: Vec3, velocity: Vec3, _mass: f32) -> ForceOutput {
        let delta = position - self.anchor;
        let length = delta.length();
        
        if length < 1e-6 {
            return ForceOutput::default();
        }
        
        let direction = delta / length;
        let extension = length - self.rest_length;
        
        // Hooke's law + damping
        let spring_force = -self.stiffness * extension * direction;
        let damping_force = -self.damping * velocity.dot(direction) * direction;
        
        ForceOutput::force(spring_force + damping_force)
    }
}

/// Drag force proportional to velocity
#[derive(Debug, Clone, Copy)]
pub struct Drag {
    /// Linear drag coefficient
    pub k1: f32,
    /// Quadratic drag coefficient  
    pub k2: f32,
}

impl Drag {
    pub fn new(k1: f32, k2: f32) -> Self {
        Self { k1, k2 }
    }
    
    pub fn linear(k: f32) -> Self {
        Self::new(k, 0.0)
    }
    
    pub fn quadratic(k: f32) -> Self {
        Self::new(0.0, k)
    }
    
    pub fn air() -> Self {
        Self::new(0.1, 0.01)
    }
    
    pub fn water() -> Self {
        Self::new(5.0, 1.0)
    }
}

impl ForceGenerator for Drag {
    fn apply(&self, _position: Vec3, velocity: Vec3, _mass: f32) -> ForceOutput {
        let speed = velocity.length();
        
        if speed < 1e-6 {
            return ForceOutput::default();
        }
        
        let direction = velocity / speed;
        let drag_mag = self.k1 * speed + self.k2 * speed * speed;
        
        ForceOutput::force(-drag_mag * direction)
    }
}

/// Point attractor/repulsor
#[derive(Debug, Clone, Copy)]
pub struct PointForce {
    pub position: Vec3,
    pub strength: f32, // positive = attract, negative = repel
    pub falloff: Falloff,
}

#[derive(Debug, Clone, Copy)]
pub enum Falloff {
    Constant,
    Linear,
    InverseSquare,
}

impl PointForce {
    pub fn attractor(position: Vec3, strength: f32) -> Self {
        Self { position, strength, falloff: Falloff::InverseSquare }
    }
    
    pub fn repulsor(position: Vec3, strength: f32) -> Self {
        Self { position, strength: -strength, falloff: Falloff::InverseSquare }
    }
}

impl ForceGenerator for PointForce {
    fn apply(&self, position: Vec3, _velocity: Vec3, _mass: f32) -> ForceOutput {
        let delta = self.position - position;
        let dist_sq = delta.length_squared();
        
        if dist_sq < 1e-6 {
            return ForceOutput::default();
        }
        
        let dist = dist_sq.sqrt();
        let direction = delta / dist;
        
        let magnitude = match self.falloff {
            Falloff::Constant => self.strength,
            Falloff::Linear => self.strength / dist,
            Falloff::InverseSquare => self.strength / dist_sq,
        };
        
        ForceOutput::force(magnitude * direction)
    }
}

/// Buoyancy force for fluid simulation
#[derive(Debug, Clone, Copy)]
pub struct Buoyancy {
    pub fluid_surface_y: f32,
    pub fluid_density: f32,
    pub volume: f32,
}

impl Buoyancy {
    pub fn new(fluid_surface_y: f32, fluid_density: f32, volume: f32) -> Self {
        Self { fluid_surface_y, fluid_density, volume }
    }
    
    pub fn water(surface_y: f32, volume: f32) -> Self {
        Self::new(surface_y, 1000.0, volume)
    }
}

impl ForceGenerator for Buoyancy {
    fn apply(&self, position: Vec3, _velocity: Vec3, _mass: f32) -> ForceOutput {
        let depth = self.fluid_surface_y - position.y;
        
        if depth <= 0.0 {
            return ForceOutput::default();
        }
        
        // Simplified: assume sphere-like submersion
        let submerged_ratio = (depth / 2.0).min(1.0);
        let buoyancy = self.fluid_density * self.volume * submerged_ratio * 9.81;
        
        ForceOutput::force(Vec3::new(0.0, buoyancy, 0.0))
    }
}

/// Collection of force generators
pub struct ForceRegistry {
    generators: Vec<Box<dyn ForceGenerator>>,
}

impl ForceRegistry {
    pub fn new() -> Self {
        Self { generators: Vec::new() }
    }
    
    pub fn add<F: ForceGenerator + 'static>(&mut self, generator: F) {
        self.generators.push(Box::new(generator));
    }
    
    pub fn apply_all(&self, position: Vec3, velocity: Vec3, mass: f32) -> ForceOutput {
        let mut total = ForceOutput::default();
        
        for gen in &self.generators {
            if gen.is_active() {
                let output = gen.apply(position, velocity, mass);
                total.force += output.force;
                total.torque += output.torque;
            }
        }
        
        total
    }
    
    pub fn clear(&mut self) {
        self.generators.clear();
    }
    
    pub fn len(&self) -> usize {
        self.generators.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }
}

impl Default for ForceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gravity_earth() {
        let g = Gravity::earth();
        let out = g.apply(Vec3::ZERO, Vec3::ZERO, 10.0);
        assert!((out.force.y - (-98.1)).abs() < 0.01);
    }

    #[test]
    fn test_gravity_moon() {
        let g = Gravity::moon();
        let out = g.apply(Vec3::ZERO, Vec3::ZERO, 10.0);
        assert!((out.force.y - (-16.2)).abs() < 0.01);
    }

    #[test]
    fn test_spring_at_rest() {
        let s = Spring::new(Vec3::ZERO, 1.0, 100.0, 10.0);
        let out = s.apply(Vec3::X, Vec3::ZERO, 1.0);
        assert!(out.force.length() < 0.01);
    }

    #[test]
    fn test_spring_extended() {
        let s = Spring::new(Vec3::ZERO, 1.0, 100.0, 0.0);
        let out = s.apply(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, 1.0);
        assert!(out.force.x < 0.0); // pulls back
        assert!((out.force.x - (-100.0)).abs() < 0.01);
    }

    #[test]
    fn test_drag_linear() {
        let d = Drag::linear(1.0);
        let out = d.apply(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), 1.0);
        assert!((out.force.x - (-10.0)).abs() < 0.01);
    }

    #[test]
    fn test_drag_quadratic() {
        let d = Drag::quadratic(1.0);
        let out = d.apply(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), 1.0);
        assert!((out.force.x - (-100.0)).abs() < 0.01);
    }

    #[test]
    fn test_point_attractor() {
        let p = PointForce::attractor(Vec3::ZERO, 100.0);
        let out = p.apply(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, 1.0);
        assert!(out.force.x < 0.0); // pulls toward origin
    }

    #[test]
    fn test_buoyancy_underwater() {
        let b = Buoyancy::water(10.0, 1.0);
        let out = b.apply(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO, 1.0);
        assert!(out.force.y > 0.0); // pushes up
    }

    #[test]
    fn test_buoyancy_above_water() {
        let b = Buoyancy::water(0.0, 1.0);
        let out = b.apply(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO, 1.0);
        assert!(out.force.length() < 0.01);
    }

    #[test]
    fn test_force_registry() {
        let mut reg = ForceRegistry::new();
        reg.add(Gravity::earth());
        reg.add(Drag::air());
        assert_eq!(reg.len(), 2);
        
        let out = reg.apply_all(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), 1.0);
        assert!(out.force.y < 0.0); // gravity
    }
}
