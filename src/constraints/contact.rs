// Contact constraint for collision response

use glam::Vec3;
use crate::collision::contact::ContactManifold;

/// Contact constraint for solving collisions
#[derive(Debug, Clone)]
pub struct ContactConstraint {
    pub body_a: u32,
    pub body_b: u32,
    pub points: arrayvec::ArrayVec<ContactConstraintPoint, 4>,
    pub friction: f32,
    pub restitution: f32,
}

/// Per-point constraint data
#[derive(Debug, Clone, Copy)]
pub struct ContactConstraintPoint {
    /// Contact point relative to body A center
    pub r_a: Vec3,
    /// Contact point relative to body B center
    pub r_b: Vec3,
    /// Contact normal (A to B)
    pub normal: Vec3,
    /// Tangent directions for friction
    pub tangent1: Vec3,
    pub tangent2: Vec3,
    /// Penetration depth
    pub depth: f32,
    /// Effective mass for normal impulse
    pub normal_mass: f32,
    /// Effective mass for tangent impulses
    pub tangent_mass1: f32,
    pub tangent_mass2: f32,
    /// Accumulated impulses for warm starting
    pub normal_impulse: f32,
    pub tangent_impulse1: f32,
    pub tangent_impulse2: f32,
    /// Bias velocity for position correction
    pub bias: f32,
}

impl ContactConstraint {
    /// Create constraint from manifold
    pub fn from_manifold(
        manifold: &ContactManifold,
        pos_a: Vec3,
        pos_b: Vec3,
        inv_mass_a: f32,
        inv_mass_b: f32,
        inv_inertia_a: Vec3,
        inv_inertia_b: Vec3,
        friction: f32,
        restitution: f32,
    ) -> Self {
        let mut points = arrayvec::ArrayVec::new();
        
        for cp in &manifold.points {
            let r_a = cp.point_a - pos_a;
            let r_b = cp.point_b - pos_b;
            let normal = cp.normal;
            
            // Build tangent basis
            let (tangent1, tangent2) = build_tangent_basis(normal);
            
            // Compute effective masses
            let normal_mass = compute_effective_mass(
                normal, r_a, r_b, inv_mass_a, inv_mass_b, inv_inertia_a, inv_inertia_b
            );
            let tangent_mass1 = compute_effective_mass(
                tangent1, r_a, r_b, inv_mass_a, inv_mass_b, inv_inertia_a, inv_inertia_b
            );
            let tangent_mass2 = compute_effective_mass(
                tangent2, r_a, r_b, inv_mass_a, inv_mass_b, inv_inertia_a, inv_inertia_b
            );
            
            points.push(ContactConstraintPoint {
                r_a,
                r_b,
                normal,
                tangent1,
                tangent2,
                depth: cp.depth,
                normal_mass: if normal_mass > 0.0 { 1.0 / normal_mass } else { 0.0 },
                tangent_mass1: if tangent_mass1 > 0.0 { 1.0 / tangent_mass1 } else { 0.0 },
                tangent_mass2: if tangent_mass2 > 0.0 { 1.0 / tangent_mass2 } else { 0.0 },
                normal_impulse: cp.normal_impulse,
                tangent_impulse1: 0.0,
                tangent_impulse2: 0.0,
                bias: 0.0,
            });
        }
        
        Self {
            body_a: manifold.body_a,
            body_b: manifold.body_b,
            points,
            friction,
            restitution,
        }
    }
    
    /// Apply warm starting impulses
    pub fn warm_start(&self) -> Vec<ImpulseResult> {
        let mut results = Vec::new();
        
        for p in &self.points {
            let impulse = p.normal * p.normal_impulse 
                + p.tangent1 * p.tangent_impulse1 
                + p.tangent2 * p.tangent_impulse2;
            
            results.push(ImpulseResult {
                body_a: self.body_a,
                body_b: self.body_b,
                impulse,
                r_a: p.r_a,
                r_b: p.r_b,
            });
        }
        
        results
    }
    
    /// Solve velocity constraints (sequential impulse)
    pub fn solve_velocity(
        &mut self,
        vel_a: Vec3,
        ang_vel_a: Vec3,
        vel_b: Vec3,
        ang_vel_b: Vec3,
    ) -> Vec<ImpulseResult> {
        let mut results = Vec::new();
        
        for p in &mut self.points {
            // Relative velocity at contact
            let vel_a_at_contact = vel_a + ang_vel_a.cross(p.r_a);
            let vel_b_at_contact = vel_b + ang_vel_b.cross(p.r_b);
            let rel_vel = vel_b_at_contact - vel_a_at_contact;
            
            // Normal impulse
            let vn = rel_vel.dot(p.normal);
            let lambda_n = -p.normal_mass * (vn + p.bias);
            let new_impulse = (p.normal_impulse + lambda_n).max(0.0);
            let impulse_n = new_impulse - p.normal_impulse;
            p.normal_impulse = new_impulse;
            
            // Friction impulses
            let max_friction = self.friction * p.normal_impulse;
            
            let vt1 = rel_vel.dot(p.tangent1);
            let lambda_t1 = -p.tangent_mass1 * vt1;
            let new_t1 = (p.tangent_impulse1 + lambda_t1).clamp(-max_friction, max_friction);
            let impulse_t1 = new_t1 - p.tangent_impulse1;
            p.tangent_impulse1 = new_t1;
            
            let vt2 = rel_vel.dot(p.tangent2);
            let lambda_t2 = -p.tangent_mass2 * vt2;
            let new_t2 = (p.tangent_impulse2 + lambda_t2).clamp(-max_friction, max_friction);
            let impulse_t2 = new_t2 - p.tangent_impulse2;
            p.tangent_impulse2 = new_t2;
            
            let impulse = p.normal * impulse_n + p.tangent1 * impulse_t1 + p.tangent2 * impulse_t2;
            
            results.push(ImpulseResult {
                body_a: self.body_a,
                body_b: self.body_b,
                impulse,
                r_a: p.r_a,
                r_b: p.r_b,
            });
        }
        
        results
    }
    
    /// Solve position constraints (pseudo-velocity)
    pub fn solve_position(
        &self,
        _pos_a: Vec3,
        _pos_b: Vec3,
        inv_mass_a: f32,
        inv_mass_b: f32,
    ) -> PositionCorrection {
        let mut correction_a = Vec3::ZERO;
        let mut correction_b = Vec3::ZERO;
        
        const SLOP: f32 = 0.005; // Allowed penetration
        const BAUMGARTE: f32 = 0.2; // Position correction factor
        
        for p in &self.points {
            let separation = p.depth - SLOP;
            if separation < 0.0 {
                continue;
            }
            
            let correction = BAUMGARTE * separation;
            let total_inv_mass = inv_mass_a + inv_mass_b;
            
            if total_inv_mass > 0.0 {
                correction_a -= p.normal * correction * (inv_mass_a / total_inv_mass);
                correction_b += p.normal * correction * (inv_mass_b / total_inv_mass);
            }
        }
        
        PositionCorrection { correction_a, correction_b }
    }
    
    /// Setup bias for restitution
    pub fn setup_velocity_bias(&mut self, rel_vel_n: f32) {
        const VELOCITY_THRESHOLD: f32 = 1.0;
        
        for p in &mut self.points {
            if rel_vel_n < -VELOCITY_THRESHOLD {
                p.bias = -self.restitution * rel_vel_n;
            }
        }
    }
}

/// Result of impulse application
#[derive(Debug, Clone, Copy)]
pub struct ImpulseResult {
    pub body_a: u32,
    pub body_b: u32,
    pub impulse: Vec3,
    pub r_a: Vec3,
    pub r_b: Vec3,
}

/// Position correction result
#[derive(Debug, Clone, Copy)]
pub struct PositionCorrection {
    pub correction_a: Vec3,
    pub correction_b: Vec3,
}

fn build_tangent_basis(normal: Vec3) -> (Vec3, Vec3) {
    let tangent1 = if normal.x.abs() > 0.9 {
        normal.cross(Vec3::Y).normalize()
    } else {
        normal.cross(Vec3::X).normalize()
    };
    let tangent2 = normal.cross(tangent1);
    (tangent1, tangent2)
}

fn compute_effective_mass(
    dir: Vec3,
    r_a: Vec3,
    r_b: Vec3,
    inv_mass_a: f32,
    inv_mass_b: f32,
    inv_inertia_a: Vec3,
    inv_inertia_b: Vec3,
) -> f32 {
    let rn_a = r_a.cross(dir);
    let rn_b = r_b.cross(dir);
    
    inv_mass_a + inv_mass_b
        + rn_a.x * rn_a.x * inv_inertia_a.x
        + rn_a.y * rn_a.y * inv_inertia_a.y
        + rn_a.z * rn_a.z * inv_inertia_a.z
        + rn_b.x * rn_b.x * inv_inertia_b.x
        + rn_b.y * rn_b.y * inv_inertia_b.y
        + rn_b.z * rn_b.z * inv_inertia_b.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::contact::{ContactPoint, ContactManifold};

    fn make_manifold() -> ContactManifold {
        let mut m = ContactManifold::new(0, 1);
        m.add_point(ContactPoint::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.1, 0.0),
            Vec3::Y,
            0.1,
        ));
        m
    }

    #[test]
    fn test_from_manifold() {
        let m = make_manifold();
        let c = ContactConstraint::from_manifold(
            &m, Vec3::ZERO, Vec3::Y,
            1.0, 1.0, Vec3::ONE, Vec3::ONE,
            0.5, 0.3
        );
        assert_eq!(c.body_a, 0);
        assert_eq!(c.body_b, 1);
        assert_eq!(c.points.len(), 1);
    }

    #[test]
    fn test_warm_start() {
        let m = make_manifold();
        let c = ContactConstraint::from_manifold(
            &m, Vec3::ZERO, Vec3::Y,
            1.0, 1.0, Vec3::ONE, Vec3::ONE,
            0.5, 0.3
        );
        let impulses = c.warm_start();
        assert_eq!(impulses.len(), 1);
    }

    #[test]
    fn test_solve_velocity() {
        let m = make_manifold();
        let mut c = ContactConstraint::from_manifold(
            &m, Vec3::ZERO, Vec3::Y,
            1.0, 1.0, Vec3::ONE, Vec3::ONE,
            0.5, 0.0
        );
        let impulses = c.solve_velocity(
            Vec3::ZERO, Vec3::ZERO,
            Vec3::new(0.0, -10.0, 0.0), Vec3::ZERO
        );
        assert_eq!(impulses.len(), 1);
        assert!(impulses[0].impulse.y > 0.0);
    }

    #[test]
    fn test_solve_position() {
        let m = make_manifold();
        let c = ContactConstraint::from_manifold(
            &m, Vec3::ZERO, Vec3::Y,
            1.0, 1.0, Vec3::ONE, Vec3::ONE,
            0.5, 0.3
        );
        let corr = c.solve_position(Vec3::ZERO, Vec3::Y, 1.0, 1.0);
        assert!(corr.correction_a.y < 0.0 || corr.correction_b.y > 0.0);
    }

    #[test]
    fn test_tangent_basis() {
        let (t1, t2) = build_tangent_basis(Vec3::Y);
        assert!(t1.dot(Vec3::Y).abs() < 0.01);
        assert!(t2.dot(Vec3::Y).abs() < 0.01);
        assert!(t1.dot(t2).abs() < 0.01);
    }

    #[test]
    fn test_friction_clamping() {
        let m = make_manifold();
        let mut c = ContactConstraint::from_manifold(
            &m, Vec3::ZERO, Vec3::Y,
            1.0, 1.0, Vec3::ONE, Vec3::ONE,
            0.5, 0.0
        );
        // Set normal impulse manually
        c.points[0].normal_impulse = 10.0;
        
        let impulses = c.solve_velocity(
            Vec3::ZERO, Vec3::ZERO,
            Vec3::new(100.0, 0.0, 0.0), Vec3::ZERO
        );
        // Tangent impulse should be clamped
        let tangent_mag = (impulses[0].impulse.x.powi(2) + impulses[0].impulse.z.powi(2)).sqrt();
        assert!(tangent_mag <= 5.1); // friction * normal_impulse + tolerance
    }

    #[test]
    fn test_restitution_bias() {
        let m = make_manifold();
        let mut c = ContactConstraint::from_manifold(
            &m, Vec3::ZERO, Vec3::Y,
            1.0, 1.0, Vec3::ONE, Vec3::ONE,
            0.5, 0.8
        );
        c.setup_velocity_bias(-10.0);
        assert!(c.points[0].bias > 0.0);
    }
}
