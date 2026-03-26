//! Material properties for physics objects.

/// Material properties controlling friction and restitution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Friction coefficient (0 = frictionless, 1 = very sticky).
    pub friction: f32,
    /// Restitution/bounciness (0 = no bounce, 1 = perfectly elastic).
    pub restitution: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            friction: 0.5,
            restitution: 0.3,
        }
    }
}

impl Material {
    /// Create a new material.
    pub fn new(friction: f32, restitution: f32) -> Self {
        Self {
            friction: friction.clamp(0.0, 1.0),
            restitution: restitution.clamp(0.0, 1.0),
        }
    }

    /// Rubber-like material (high friction, bouncy).
    pub fn rubber() -> Self {
        Self {
            friction: 0.9,
            restitution: 0.8,
        }
    }

    /// Steel-like material (low friction, low bounce).
    pub fn steel() -> Self {
        Self {
            friction: 0.4,
            restitution: 0.1,
        }
    }

    /// Ice-like material (very low friction).
    pub fn ice() -> Self {
        Self {
            friction: 0.05,
            restitution: 0.1,
        }
    }

    /// Wood-like material.
    pub fn wood() -> Self {
        Self {
            friction: 0.6,
            restitution: 0.3,
        }
    }
}

/// Combine friction coefficients from two materials.
pub fn combine_friction(a: f32, b: f32) -> f32 {
    // Geometric mean - common approach
    (a * b).sqrt()
}

/// Combine restitution coefficients from two materials.
pub fn combine_restitution(a: f32, b: f32) -> f32 {
    // Maximum - allows bouncy objects to bounce on anything
    a.max(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let m = Material::default();
        assert!((m.friction - 0.5).abs() < 0.001);
        assert!((m.restitution - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_rubber() {
        let m = Material::rubber();
        assert!(m.friction > 0.8);
        assert!(m.restitution > 0.7);
    }

    #[test]
    fn test_ice() {
        let m = Material::ice();
        assert!(m.friction < 0.1);
    }

    #[test]
    fn test_combine_friction() {
        let f = combine_friction(0.25, 1.0);
        assert!((f - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_combine_restitution() {
        let r = combine_restitution(0.2, 0.8);
        assert!((r - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_clamp() {
        let m = Material::new(2.0, -1.0);
        assert_eq!(m.friction, 1.0);
        assert_eq!(m.restitution, 0.0);
    }
}
