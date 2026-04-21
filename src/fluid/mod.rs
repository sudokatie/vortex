//! Fluid simulation module using Smoothed Particle Hydrodynamics (SPH).
//!
//! This module provides components for SPH-based fluid simulation:
//!
//! - `kernel`: Smoothing kernel functions (Poly6, Spiky, Viscosity)
//! - `particle`: Fluid particles and spatial hash grid for neighbor queries
//! - `solver`: SPH solver for fluid dynamics
//! - `boundary`: Boundary conditions for constraining fluid particles
//! - `coupling`: Fluid-rigid body coupling for two-way interaction

pub mod boundary;
pub mod coupling;
pub mod kernel;
pub mod particle;
pub mod solver;

pub use kernel::{
    Kernels,
    poly6_value,
    poly6_value_2d,
    spiky_gradient,
    spiky_gradient_2d,
    viscosity_laplacian,
    viscosity_laplacian_2d,
};

pub use particle::{FluidParticle, ParticleGrid};

pub use solver::{FluidWorld, SPHSolver};

pub use boundary::{Boundary, BoxBoundary, PlaneBoundary, RigidBodyBoundary};

pub use coupling::{
    FluidCouplingParams, FluidForceOutput,
    apply_boundary_forces_to_particles, apply_fluid_forces_to_body,
    compute_buoyancy, compute_drag, estimate_volume,
};
