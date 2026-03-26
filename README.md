# Vortex

Physics engine with rigid body dynamics and collision detection.

## Features

- **Rigid Body Simulation**: Forces, torques, impulses with semi-implicit Euler integration
- **Collision Shapes**: Sphere, box, and capsule primitives with GJK/EPA support
- **Material Properties**: Friction and restitution with preset materials
- **Physics World**: Manage bodies with automatic gravity and stepping

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

## Roadmap

- [x] Basic rigid body dynamics
- [x] Collision shapes (sphere, box, capsule)
- [x] Physics world stepping
- [ ] GJK/EPA collision detection
- [ ] Contact constraint solver
- [ ] Joints (distance, ball, hinge)
- [ ] Island-based sleeping
- [ ] Broad-phase acceleration

## License

MIT
