//! Fluid simulation module using Smoothed Particle Hydrodynamics (SPH).
//!
//! This module provides components for SPH-based fluid simulation:
//!
//! - `kernel`: Smoothing kernel functions (Poly6, Spiky, Viscosity)

pub mod kernel;

pub use kernel::{
    Kernels,
    poly6_value,
    poly6_value_2d,
    spiky_gradient,
    spiky_gradient_2d,
    viscosity_laplacian,
    viscosity_laplacian_2d,
};
