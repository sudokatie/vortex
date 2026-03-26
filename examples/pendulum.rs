//! Example: Simple pendulum simulation.

use vortex::prelude::*;

fn main() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Create a dynamic sphere (the pendulum bob)
    let mut bob = RigidBody::new(
        CollisionShape::sphere(0.2),
        1.0,
        BodyType::Dynamic,
    );
    // Start offset to the side
    bob.set_position(Vec3::new(2.0, 0.0, 0.0));
    let bob_handle = world.add_body(bob);

    println!("Pendulum simulation (gravity only, no constraint yet)");
    println!("The bob will fall under gravity.\n");

    // Simulate for 2 seconds
    let dt = 1.0 / 60.0;
    for frame in 0..120 {
        world.step(dt);

        if frame % 20 == 0 {
            let bob = world.get_body(bob_handle).unwrap();
            println!(
                "Frame {:3}: pos=({:6.2}, {:6.2}, {:6.2}) vel=({:6.2}, {:6.2}, {:6.2})",
                frame,
                bob.position.x, bob.position.y, bob.position.z,
                bob.linear_velocity.x, bob.linear_velocity.y, bob.linear_velocity.z,
            );
        }
    }

    println!("\nDone! (Distance joint will constrain to pendulum motion once implemented)");
}
