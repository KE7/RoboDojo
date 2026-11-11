"""Rigid bodies nested inside another enabled rigid body, as Isaac Sim 5.1 treated them.

USD Physics requires a rigid body below another enabled rigid body to reset its xform stack. Some RoboDojo assets
(the Plug In Charger power strip) nest bodies without it. Isaac Sim 5.1's PhysX rejected those inner bodies, logging
"missing xformstack reset when child of another enabled rigid body", and their colliders became part of the outer body.
Isaac Sim 6.2 builds them, which leaves the outer body without colliders: it sinks through the table. Removing the inner
bodies' RigidBodyAPI reproduces 5.1's behaviour.
"""

from pxr import Usd, UsdGeom, UsdPhysics


def drop_nested_rigid_bodies(root: Usd.Prim, root_is_body: bool = False) -> list[str]:
    """Remove RigidBodyAPI from bodies under an enabled rigid body that do not reset their xform stack. Their paths.

    ``root_is_body``: the root is (or is about to be made) an enabled rigid body, as RoboDojo's ``RigidObject`` makes it
    after loading the asset.
    """

    def enabled_body(prim: Usd.Prim) -> bool:
        if root_is_body and prim == root:
            return True
        return prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False

    dropped = []
    for prim in Usd.PrimRange(root):
        if prim == root or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        parent = prim.GetParent()
        while parent and parent != root.GetParent():
            if enabled_body(parent):
                break
            parent = parent.GetParent()
        else:
            continue
        if not parent or UsdGeom.Xformable(prim).GetResetXformStack():
            continue
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        dropped.append(str(prim.GetPath()))
    return dropped
