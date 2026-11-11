"""``create_mdl_material``, as Isaac Sim 5.1 shipped it in ``isaacsim.replicator.behavior.utils.scene_utils``.

Isaac Sim 6.2 removed the helper from that module; RoboDojo's tables, grounds and fluids still create their MDL
materials this way.
"""

import omni.kit.commands
import omni.usd
from pxr import Sdf, UsdShade


def create_mdl_material(mtl_url: str, mtl_name: str, mtl_path: str) -> UsdShade.Material:
    """Creates an MDL material at the given path with the specified URL and name."""
    omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=mtl_url, mtl_name=mtl_name, mtl_path=mtl_path)
    stage = omni.usd.get_context().get_stage()
    mtl_prim = stage.GetPrimAtPath(mtl_path)
    shader = UsdShade.Shader(omni.usd.get_shader_from_material(mtl_prim, get_prim=True))
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool)
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2)
    shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float)
    return UsdShade.Material(mtl_prim)
