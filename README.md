# Vortex

Physics engine with rigid body dynamics, fluid simulation, and collision detection.

## Features

- **Rigid Body Simulation**: Forces, torques, impulses with semi-implicit Euler integration
- **Soft Body Dynamics**: Deformable objects using mass-spring systems
- **SPH Fluid Simulation**: Particle-based fluids with spatial hashing for neighbor queries
- **Collision Detection**: GJK/EPA for narrow-phase, with SAP, BVH, and spatial hash broad-phase
- **Continuous Collision Detection**: Sweep tests and time-of-impact calculations for fast-moving objects
- **Collision Shapes**: Sphere, box, capsule, convex hull, and triangle mesh colliders
- **Contact Solver**: Iterative constraint solver with warm starting
- **Joints**: Distance, ball, and hinge constraints
- **Island-based Sleeping**: Automatic deactivation of resting body groups
- **Material Properties**: Friction and restitution with preset materials
- **2D Physics**: Optional 2D module via `dim2` feature flag

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
vortex = "0.1"
```

## Quick Start

```rust
use vortex::prelude::*;

// Create a physics world
let mut world = PhysicsWorld::new();
world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

// Add a dynamic sphere
let mut ball = RigidBody::new(
    CollisionShape::sphere(0.5),
    1.0,
    BodyType::Dynamic,
);
ball.set_position(Vec3::new(0.0, 10.0, 0.0));
let ball_handle = world.add_body(ball);

// Add a static floor
let mut floor = RigidBody::new(
    CollisionShape::cube(Vec3::new(10.0, 0.1, 10.0)),
    1.0,
    BodyType::Static,
);
world.add_body(floor);

// Simulate at 60 FPS
for _ in 0..60 {
    world.step(1.0 / 60.0);
}

// Check the ball's position
let ball = world.get_body(ball_handle).unwrap();
println!("Ball position: {:?}", ball.position);
```

## Collision Shapes

```rust
// Sphere with radius
let sphere = CollisionShape::sphere(1.0);

// Box with half-extents
let cube = CollisionShape::cube(Vec3::new(0.5, 0.5, 0.5));

// Capsule with radius and half-height
let capsule = CollisionShape::capsule(0.5, 1.0);

// Convex hull from vertices (uses QuickHull algorithm)
let vertices = vec![
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(-1.0, -0.5, -0.5),
    Vec3::new(1.0, -0.5, -0.5),
    Vec3::new(0.0, -0.5, 1.0),
];
let convex = CollisionShape::convex_hull(vertices);

// Triangle mesh from vertices and indices
let mesh_vertices = vec![
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.5, 1.0, 0.0),
    Vec3::new(0.5, 0.5, 1.0),
];
let indices = vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];
let mesh = CollisionShape::from_vertices_and_indices(&mesh_vertices, &indices);

// Decompose mesh into convex hulls for faster collision
let convex_parts = mesh.convex_decompose(2); // depth parameter
```

## Materials

```rust
// Custom material
let mat = Material::new(0.6, 0.3); // friction, restitution

// Presets
let rubber = Material::rubber();  // bouncy
let steel = Material::steel();    // rigid
let ice = Material::ice();        // slippery
```

## Body Types

- `BodyType::Dynamic` - Fully simulated by physics
- `BodyType::Kinematic` - Moved by code, affects dynamic bodies
- `BodyType::Static` - Immovable, infinite mass

## Fluid Simulation

```rust
use vortex::prelude::*;
use vortex::fluid::{FluidWorld, FluidParticle, BoxBoundary};

// Create a fluid world with smoothing radius
let mut fluid = FluidWorld::new(0.5);

// Configure the solver
fluid.solver_mut().gravity = Vec3::new(0.0, -9.81, 0.0);
fluid.solver_mut().rest_density = 1000.0;
fluid.solver_mut().viscosity_coeff = 200.0;

// Add a container boundary
fluid.solver_mut().add_box_boundary(BoxBoundary::new(
    Vec3::new(-2.0, 0.0, -2.0),
    Vec3::new(2.0, 4.0, 2.0),
));

// Spawn particles in a grid
for x in 0..10 {
    for y in 0..10 {
        for z in 0..10 {
            let pos = Vec3::new(
                x as f32 * 0.1 - 0.5,
                y as f32 * 0.1 + 1.0,
                z as f32 * 0.1 - 0.5,
            );
            fluid.add_particle(FluidParticle::new(pos, 1.0));
        }
    }
}

// Simulate
for _ in 0..1000 {
    fluid.step(0.001);
}

// Access particle positions for rendering
for particle in fluid.particles() {
    println!("{:?}", particle.position);
}
```

## Roadmap

- [x] Basic rigid body dynamics
- [x] Collision shapes (sphere, box, capsule)
- [x] Physics world stepping
- [x] GJK/EPA collision detection
- [x] Contact constraint solver
- [x] Joints (distance, ball, hinge)
- [x] Island-based sleeping
- [x] Broad-phase acceleration
- [x] Continuous collision detection (CCD)
- [x] Convex hull and mesh colliders
- [ ] Fluid-rigid body coupling
- [ ] GPU acceleration

## License

MIT

---

*Katie*
