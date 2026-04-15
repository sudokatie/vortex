//! SPH kernel functions for fluid simulation.
//!
//! This module provides smoothing kernel functions used in Smoothed Particle
//! Hydrodynamics (SPH) simulations. Each kernel serves a specific purpose:
//!
//! - **Poly6**: Smooth, non-negative kernel used for density estimation
//! - **Spiky**: Sharp gradient kernel used for pressure forces (avoids particle clumping)
//! - **Viscosity**: Laplacian kernel used for viscosity forces

use glam::{Vec2, Vec3};
use std::f64::consts::PI;

/// Precomputed kernel constants for a given smoothing radius h.
///
/// Computing kernel constants is expensive due to the powers and divisions involved.
/// This struct precomputes all h-dependent constants once for efficiency.
#[derive(Debug, Clone, Copy)]
pub struct Kernels {
    /// Smoothing radius
    pub h: f64,
    /// Precomputed: 315 / (64 * pi * h^9) for 3D Poly6
    poly6_3d: f64,
    /// Precomputed: -45 / (pi * h^6) for 3D Spiky gradient
    spiky_3d: f64,
    /// Precomputed: 45 / (pi * h^6) for 3D Viscosity Laplacian
    visc_3d: f64,
    /// Precomputed: 4 / (pi * h^8) for 2D Poly6
    poly6_2d: f64,
    /// Precomputed: -30 / (pi * h^5) for 2D Spiky gradient
    spiky_2d: f64,
    /// Precomputed: 40 / (pi * h^5) for 2D Viscosity Laplacian
    visc_2d: f64,
    /// h^2 for quick comparison
    h2: f64,
}

impl Kernels {
    /// Create a new Kernels instance with precomputed constants for the given smoothing radius.
    ///
    /// # Panics
    /// Panics if `h <= 0.0`.
    pub fn new(h: f64) -> Self {
        assert!(h > 0.0, "Smoothing radius h must be positive");

        let h2 = h * h;
        let h3 = h2 * h;
        let h5 = h3 * h2;
        let h6 = h3 * h3;
        let h8 = h6 * h2;
        let h9 = h6 * h3;

        Self {
            h,
            poly6_3d: 315.0 / (64.0 * PI * h9),
            spiky_3d: -45.0 / (PI * h6),
            visc_3d: 45.0 / (PI * h6),
            poly6_2d: 4.0 / (PI * h8),
            spiky_2d: -30.0 / (PI * h5),
            visc_2d: 40.0 / (PI * h5),
            h2,
        }
    }

    /// Poly6 kernel value (3D).
    ///
    /// Used for density estimation. Smooth and non-negative.
    #[inline]
    pub fn poly6(&self, r: f64) -> f64 {
        if r >= self.h || r < 0.0 {
            return 0.0;
        }
        let diff = self.h2 - r * r;
        self.poly6_3d * diff * diff * diff
    }

    /// Spiky kernel gradient (3D).
    ///
    /// Used for pressure force computation. Sharp gradient avoids particle clumping.
    #[inline]
    pub fn spiky_grad(&self, r_vec: Vec3) -> Vec3 {
        let r = r_vec.length() as f64;
        if r >= self.h || r < 1e-10 {
            return Vec3::ZERO;
        }
        let diff = self.h - r;
        let coeff = self.spiky_3d * diff * diff / r;
        r_vec * coeff as f32
    }

    /// Viscosity kernel Laplacian (3D).
    ///
    /// Used for viscosity force computation.
    #[inline]
    pub fn visc_laplacian(&self, r: f64) -> f64 {
        if r >= self.h || r < 0.0 {
            return 0.0;
        }
        self.visc_3d * (self.h - r)
    }

    /// Poly6 kernel value (2D).
    #[inline]
    pub fn poly6_2d(&self, r: f64) -> f64 {
        if r >= self.h || r < 0.0 {
            return 0.0;
        }
        let diff = self.h2 - r * r;
        self.poly6_2d * diff * diff * diff
    }

    /// Spiky kernel gradient (2D).
    #[inline]
    pub fn spiky_grad_2d(&self, r_vec: Vec2) -> Vec2 {
        let r = r_vec.length() as f64;
        if r >= self.h || r < 1e-10 {
            return Vec2::ZERO;
        }
        let diff = self.h - r;
        let coeff = self.spiky_2d * diff * diff / r;
        r_vec * coeff as f32
    }

    /// Viscosity kernel Laplacian (2D).
    #[inline]
    pub fn visc_laplacian_2d(&self, r: f64) -> f64 {
        if r >= self.h || r < 0.0 {
            return 0.0;
        }
        self.visc_2d * (self.h - r)
    }
}

/// Poly6 kernel value (3D).
///
/// W(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
///
/// Returns 0.0 when r >= h (outside support radius).
#[inline]
pub fn poly6_value(r: f64, h: f64) -> f64 {
    if r >= h || r < 0.0 {
        return 0.0;
    }
    let h2 = h * h;
    let h9 = h2 * h2 * h2 * h2 * h;
    let diff = h2 - r * r;
    315.0 / (64.0 * PI * h9) * diff * diff * diff
}

/// Spiky kernel gradient (3D).
///
/// grad W(r, h) = -45 / (pi * h^6) * (h - r)^2 * r_hat
///
/// Returns Vec3::ZERO when r >= h or r is too small (to avoid division by zero).
#[inline]
pub fn spiky_gradient(r_vec: Vec3, h: f64) -> Vec3 {
    let r = r_vec.length() as f64;
    if r >= h || r < 1e-10 {
        return Vec3::ZERO;
    }
    let h6 = h * h * h * h * h * h;
    let diff = h - r;
    let coeff = -45.0 / (PI * h6) * diff * diff / r;
    r_vec * coeff as f32
}

/// Viscosity kernel Laplacian (3D).
///
/// laplacian W(r, h) = 45 / (pi * h^6) * (h - r)
///
/// Returns 0.0 when r >= h (outside support radius).
#[inline]
pub fn viscosity_laplacian(r: f64, h: f64) -> f64 {
    if r >= h || r < 0.0 {
        return 0.0;
    }
    let h6 = h * h * h * h * h * h;
    45.0 / (PI * h6) * (h - r)
}

/// Poly6 kernel value (2D).
///
/// W(r, h) = 4 / (pi * h^8) * (h^2 - r^2)^3
///
/// Returns 0.0 when r >= h (outside support radius).
#[inline]
pub fn poly6_value_2d(r: f64, h: f64) -> f64 {
    if r >= h || r < 0.0 {
        return 0.0;
    }
    let h2 = h * h;
    let h8 = h2 * h2 * h2 * h2;
    let diff = h2 - r * r;
    4.0 / (PI * h8) * diff * diff * diff
}

/// Spiky kernel gradient (2D).
///
/// grad W(r, h) = -30 / (pi * h^5) * (h - r)^2 * r_hat
///
/// Returns Vec2::ZERO when r >= h or r is too small.
#[inline]
pub fn spiky_gradient_2d(r_vec: Vec2, h: f64) -> Vec2 {
    let r = r_vec.length() as f64;
    if r >= h || r < 1e-10 {
        return Vec2::ZERO;
    }
    let h5 = h * h * h * h * h;
    let diff = h - r;
    let coeff = -30.0 / (PI * h5) * diff * diff / r;
    r_vec * coeff as f32
}

/// Viscosity kernel Laplacian (2D).
///
/// laplacian W(r, h) = 40 / (pi * h^5) * (h - r)
///
/// Returns 0.0 when r >= h (outside support radius).
#[inline]
pub fn viscosity_laplacian_2d(r: f64, h: f64) -> f64 {
    if r >= h || r < 0.0 {
        return 0.0;
    }
    let h5 = h * h * h * h * h;
    40.0 / (PI * h5) * (h - r)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;
    const H: f64 = 1.0;

    // ==================== Poly6 3D Tests ====================

    #[test]
    fn poly6_at_zero() {
        // Maximum value at r = 0
        let w = poly6_value(0.0, H);
        let expected = 315.0 / (64.0 * PI); // h^9 = 1, (h^2 - 0)^3 = 1
        assert!((w - expected).abs() < EPSILON, "poly6(0) = {} != {}", w, expected);
    }

    #[test]
    fn poly6_at_boundary() {
        // Zero at r = h
        let w = poly6_value(H, H);
        assert!(w.abs() < EPSILON, "poly6(h) should be 0, got {}", w);
    }

    #[test]
    fn poly6_outside_support() {
        // Zero outside support radius
        let w = poly6_value(H + 0.1, H);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn poly6_negative_r() {
        let w = poly6_value(-0.5, H);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn poly6_is_positive() {
        for i in 1..10 {
            let r = (i as f64) * 0.1;
            let w = poly6_value(r, H);
            assert!(w >= 0.0, "poly6({}) = {} should be >= 0", r, w);
        }
    }

    #[test]
    fn poly6_monotonically_decreasing() {
        let mut prev = poly6_value(0.0, H);
        for i in 1..10 {
            let r = (i as f64) * 0.1;
            let w = poly6_value(r, H);
            assert!(w <= prev, "poly6 should decrease: {} > {}", prev, w);
            prev = w;
        }
    }

    // ==================== Spiky Gradient 3D Tests ====================

    #[test]
    fn spiky_gradient_at_zero() {
        // Gradient is zero at r = 0 (avoid singularity)
        let g = spiky_gradient(Vec3::ZERO, H);
        assert_eq!(g, Vec3::ZERO);
    }

    #[test]
    fn spiky_gradient_at_boundary() {
        let g = spiky_gradient(Vec3::new(H as f32, 0.0, 0.0), H);
        assert!(g.length() < 1e-6, "spiky gradient at boundary should be ~0, got {:?}", g);
    }

    #[test]
    fn spiky_gradient_outside_support() {
        let g = spiky_gradient(Vec3::new((H + 0.1) as f32, 0.0, 0.0), H);
        assert_eq!(g, Vec3::ZERO);
    }

    #[test]
    fn spiky_gradient_points_toward_origin() {
        // Gradient should point from particle position toward origin (negative direction)
        let r_vec = Vec3::new(0.5, 0.0, 0.0);
        let g = spiky_gradient(r_vec, H);
        assert!(g.x < 0.0, "gradient x should be negative, got {}", g.x);
        assert!(g.y.abs() < 1e-10);
        assert!(g.z.abs() < 1e-10);
    }

    #[test]
    fn spiky_gradient_magnitude_increases_near_center() {
        // Spiky kernel has increasing gradient magnitude as we approach center
        let g1 = spiky_gradient(Vec3::new(0.1, 0.0, 0.0), H);
        let g2 = spiky_gradient(Vec3::new(0.5, 0.0, 0.0), H);
        assert!(
            g1.length() > g2.length(),
            "gradient at 0.1 ({}) should be > gradient at 0.5 ({})",
            g1.length(),
            g2.length()
        );
    }

    #[test]
    fn spiky_gradient_symmetry() {
        let g_pos = spiky_gradient(Vec3::new(0.5, 0.0, 0.0), H);
        let g_neg = spiky_gradient(Vec3::new(-0.5, 0.0, 0.0), H);
        assert!((g_pos.x + g_neg.x).abs() < 1e-6);
        assert!((g_pos.y - g_neg.y).abs() < 1e-6);
    }

    // ==================== Viscosity Laplacian 3D Tests ====================

    #[test]
    fn viscosity_laplacian_at_zero() {
        let lap = viscosity_laplacian(0.0, H);
        let expected = 45.0 / PI; // h^6 = 1, (h - 0) = 1
        assert!((lap - expected).abs() < EPSILON, "visc_lap(0) = {} != {}", lap, expected);
    }

    #[test]
    fn viscosity_laplacian_at_boundary() {
        let lap = viscosity_laplacian(H, H);
        assert!(lap.abs() < EPSILON, "visc_lap(h) should be 0, got {}", lap);
    }

    #[test]
    fn viscosity_laplacian_outside_support() {
        let lap = viscosity_laplacian(H + 0.1, H);
        assert_eq!(lap, 0.0);
    }

    #[test]
    fn viscosity_laplacian_negative_r() {
        let lap = viscosity_laplacian(-0.5, H);
        assert_eq!(lap, 0.0);
    }

    #[test]
    fn viscosity_laplacian_is_positive() {
        for i in 0..10 {
            let r = (i as f64) * 0.1;
            let lap = viscosity_laplacian(r, H);
            assert!(lap >= 0.0, "visc_lap({}) = {} should be >= 0", r, lap);
        }
    }

    #[test]
    fn viscosity_laplacian_linear_decrease() {
        // The viscosity Laplacian decreases linearly with r
        let lap1 = viscosity_laplacian(0.2, H);
        let lap2 = viscosity_laplacian(0.4, H);
        let lap3 = viscosity_laplacian(0.6, H);
        let diff1 = lap1 - lap2;
        let diff2 = lap2 - lap3;
        assert!((diff1 - diff2).abs() < EPSILON, "should be linear: {} != {}", diff1, diff2);
    }

    // ==================== Poly6 2D Tests ====================

    #[test]
    fn poly6_2d_at_zero() {
        let w = poly6_value_2d(0.0, H);
        let expected = 4.0 / PI; // h^8 = 1, (h^2 - 0)^3 = 1
        assert!((w - expected).abs() < EPSILON, "poly6_2d(0) = {} != {}", w, expected);
    }

    #[test]
    fn poly6_2d_at_boundary() {
        let w = poly6_value_2d(H, H);
        assert!(w.abs() < EPSILON, "poly6_2d(h) should be 0, got {}", w);
    }

    #[test]
    fn poly6_2d_outside_support() {
        let w = poly6_value_2d(H + 0.1, H);
        assert_eq!(w, 0.0);
    }

    // ==================== Spiky Gradient 2D Tests ====================

    #[test]
    fn spiky_gradient_2d_at_zero() {
        let g = spiky_gradient_2d(Vec2::ZERO, H);
        assert_eq!(g, Vec2::ZERO);
    }

    #[test]
    fn spiky_gradient_2d_at_boundary() {
        let g = spiky_gradient_2d(Vec2::new(H as f32, 0.0), H);
        assert!(g.length() < 1e-6);
    }

    #[test]
    fn spiky_gradient_2d_outside_support() {
        let g = spiky_gradient_2d(Vec2::new((H + 0.1) as f32, 0.0), H);
        assert_eq!(g, Vec2::ZERO);
    }

    #[test]
    fn spiky_gradient_2d_points_toward_origin() {
        let r_vec = Vec2::new(0.5, 0.0);
        let g = spiky_gradient_2d(r_vec, H);
        assert!(g.x < 0.0, "gradient x should be negative, got {}", g.x);
    }

    // ==================== Viscosity Laplacian 2D Tests ====================

    #[test]
    fn viscosity_laplacian_2d_at_zero() {
        let lap = viscosity_laplacian_2d(0.0, H);
        let expected = 40.0 / PI; // h^5 = 1, (h - 0) = 1
        assert!((lap - expected).abs() < EPSILON, "visc_lap_2d(0) = {} != {}", lap, expected);
    }

    #[test]
    fn viscosity_laplacian_2d_at_boundary() {
        let lap = viscosity_laplacian_2d(H, H);
        assert!(lap.abs() < EPSILON, "visc_lap_2d(h) should be 0, got {}", lap);
    }

    #[test]
    fn viscosity_laplacian_2d_outside_support() {
        let lap = viscosity_laplacian_2d(H + 0.1, H);
        assert_eq!(lap, 0.0);
    }

    // ==================== Kernels Struct Tests ====================

    #[test]
    fn kernels_matches_standalone() {
        let k = Kernels::new(H);

        // Poly6
        assert!((k.poly6(0.0) - poly6_value(0.0, H)).abs() < EPSILON);
        assert!((k.poly6(0.5) - poly6_value(0.5, H)).abs() < EPSILON);

        // Viscosity
        assert!((k.visc_laplacian(0.0) - viscosity_laplacian(0.0, H)).abs() < EPSILON);
        assert!((k.visc_laplacian(0.5) - viscosity_laplacian(0.5, H)).abs() < EPSILON);

        // Poly6 2D
        assert!((k.poly6_2d(0.0) - poly6_value_2d(0.0, H)).abs() < EPSILON);
        assert!((k.poly6_2d(0.5) - poly6_value_2d(0.5, H)).abs() < EPSILON);

        // Viscosity 2D
        assert!((k.visc_laplacian_2d(0.0) - viscosity_laplacian_2d(0.0, H)).abs() < EPSILON);
        assert!((k.visc_laplacian_2d(0.5) - viscosity_laplacian_2d(0.5, H)).abs() < EPSILON);
    }

    #[test]
    fn kernels_spiky_matches_standalone() {
        let k = Kernels::new(H);
        let r_vec3 = Vec3::new(0.5, 0.3, 0.1);
        let r_vec2 = Vec2::new(0.5, 0.3);

        let g1 = k.spiky_grad(r_vec3);
        let g2 = spiky_gradient(r_vec3, H);
        assert!((g1 - g2).length() < 1e-6);

        let g1 = k.spiky_grad_2d(r_vec2);
        let g2 = spiky_gradient_2d(r_vec2, H);
        assert!((g1 - g2).length() < 1e-6);
    }

    #[test]
    fn kernels_different_h() {
        let k1 = Kernels::new(0.5);
        let k2 = Kernels::new(2.0);

        // Poly6 at r=0: W(0,h) = 315/(64*pi*h^9) * h^6 = 315/(64*pi*h^3)
        // So w1/w2 = (h2/h1)^3
        let w1 = k1.poly6(0.0);
        let w2 = k2.poly6(0.0);
        let ratio = w1 / w2;
        let expected_ratio = (2.0_f64 / 0.5).powi(3);
        assert!((ratio - expected_ratio).abs() / expected_ratio < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Smoothing radius h must be positive")]
    fn kernels_panics_on_zero_h() {
        Kernels::new(0.0);
    }

    #[test]
    #[should_panic(expected = "Smoothing radius h must be positive")]
    fn kernels_panics_on_negative_h() {
        Kernels::new(-1.0);
    }

    // ==================== Normalization Tests ====================

    #[test]
    fn poly6_3d_normalization() {
        // Numerical integration to verify kernel integrates to ~1 over volume
        let h = 1.0;
        let n = 50;
        let dr = h / (n as f64);
        let mut integral = 0.0;

        for i in 0..n {
            let r = (i as f64 + 0.5) * dr;
            let w = poly6_value(r, h);
            // Volume element in spherical coords: 4*pi*r^2*dr
            integral += w * 4.0 * PI * r * r * dr;
        }

        assert!(
            (integral - 1.0).abs() < 0.05,
            "3D Poly6 integral = {}, expected ~1.0",
            integral
        );
    }

    #[test]
    fn poly6_2d_normalization() {
        // Numerical integration to verify kernel integrates to ~1 over area
        let h = 1.0;
        let n = 50;
        let dr = h / (n as f64);
        let mut integral = 0.0;

        for i in 0..n {
            let r = (i as f64 + 0.5) * dr;
            let w = poly6_value_2d(r, h);
            // Area element in polar coords: 2*pi*r*dr
            integral += w * 2.0 * PI * r * dr;
        }

        assert!(
            (integral - 1.0).abs() < 0.05,
            "2D Poly6 integral = {}, expected ~1.0",
            integral
        );
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn kernels_very_small_h() {
        let k = Kernels::new(0.001);
        assert!(k.poly6(0.0).is_finite());
        assert!(k.visc_laplacian(0.0).is_finite());
    }

    #[test]
    fn kernels_large_h() {
        let k = Kernels::new(1000.0);
        assert!(k.poly6(0.0).is_finite());
        assert!(k.poly6(500.0) > 0.0);
        assert!(k.visc_laplacian(0.0).is_finite());
    }

    #[test]
    fn spiky_gradient_diagonal() {
        let r_vec = Vec3::new(0.3, 0.3, 0.3);
        let g = spiky_gradient(r_vec, H);

        // Gradient should be parallel to r_vec (pointing opposite direction)
        let r_norm = r_vec.normalize();
        let g_norm = g.normalize();
        let dot = r_norm.dot(g_norm);
        assert!(
            (dot + 1.0).abs() < 1e-5,
            "gradient should be antiparallel to r_vec, dot = {}",
            dot
        );
    }
}
