"""The PhysX tensor simulation view RoboDojo's prims initialize against.

RoboDojo's prims are Isaac Sim's ``isaacsim.core.prims`` classes, which look their view up in Isaac Sim's
``SimulationManager`` themselves. Isaac Lab 2.x ran physics through that manager. Isaac Lab 3.0 runs it through its
own PhysX manager, which owns one ``warp`` view, and Isaac Sim's manager then holds none.

So under Isaac Lab 3.0 Isaac Sim's manager is given what its own ``initialize_physics`` would have created (a warp view
and a view of its tensor backend over the same stage) without its two warm-up physics steps: Isaac Lab's warp view,
and a new torch (or numpy) view of the stage Isaac Lab simulates.
"""


def get_physics_sim_view():
    from isaacsim.core.simulation_manager import SimulationManager

    view = SimulationManager.get_physics_sim_view()
    if view is not None:
        return view
    try:
        from isaaclab.sim import SimulationContext
    except ImportError:
        return None
    sim = SimulationContext.instance()
    warp_view = None if sim is None else getattr(sim, "physics_sim_view", None)
    if warp_view is None:                       # physics has not started: nothing to bind yet
        return None
    import omni.physics.tensors
    import omni.usd

    manager = SimulationManager
    if "cuda" in manager.get_physics_sim_device() and manager.get_backend() == "numpy":
        manager._backend = "torch"              # as Isaac Sim's manager does: NumPy cannot read GPU pipelines
    stage_id = omni.usd.get_context().get_stage_id()
    view = omni.physics.tensors.create_simulation_view(manager.get_backend(), stage_id=stage_id)
    view.set_subspace_roots("/")
    manager._physics_sim_view = view
    manager._physics_sim_view__warp = warp_view
    manager._warmup_needed = False
    manager._simulation_view_created = True
    return view


def configure_isaac_sim_manager(device: str) -> None:
    """Torch tensors on the simulation device in Isaac Sim's manager, before any prim exists.

    Isaac Lab 2.x's ``SimulationContext`` created Isaac Sim's simulation state with ``backend="torch"``; under Isaac Lab
    3.0 nothing does, and prims built while the manager is still on NumPy read NumPy where they are later given torch.
    """
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.set_backend("torch")
    SimulationManager.set_physics_sim_device(device)
