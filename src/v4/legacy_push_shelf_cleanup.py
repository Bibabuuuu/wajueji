"""Remove the identified historical push-test fixture from generated models.

Frozen capacity assets remain read-only. Contact classification and guards are
unchanged; the actual unwanted geometry is removed, not hidden from reporting.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

NAME = "r1_ground_push_shelf"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_generated_model(path):
    """Clean one generated XML; reject a renamed or structurally changed fixture."""
    path = Path(path)
    before = sha256(path)
    tree = ET.parse(path)
    root = tree.getroot()
    world = root.find("worldbody")
    matches = [g for g in root.iter("geom") if g.get("name") == NAME]
    if len(matches) > 1:
        raise ValueError("legacy_shelf_duplicate")
    removed = []
    if matches:
        geom = matches[0]
        if world is None or geom not in list(world):
            raise ValueError("legacy_shelf_not_world_fixture")
        if (geom.get("type") != "box" or geom.get("contype") != "1024"
                or geom.get("conaffinity") != "256"):
            raise ValueError("legacy_shelf_identity_changed")
        for key, expected in (
                ("pos", (5.13, .65, 2.86)),
                ("size", (.6, .55, .03)),
                ("quat", (1., 0., 0., 0.))):
            values = [float(x) for x in geom.get(key, "").split()]
            if len(values) != len(expected) or not all(
                    abs(a-b) <= 1e-12 for a, b in zip(values, expected)):
                raise ValueError("legacy_shelf_geometry_changed:" + key)
        for element in root.iter():
            if element is not geom and NAME in element.attrib.values():
                raise ValueError("legacy_shelf_has_reference")
        removed.append(dict(geom.attrib))
        world.remove(geom)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="unicode", xml_declaration=False)
        with path.open("a") as stream:
            stream.write("\n")
    return {
        "enabled": True, "removed_geoms": removed,
        "removed_count": len(removed), "geometry_name": NAME,
        "generated_model_before_sha256": before,
        "generated_model_after_sha256": sha256(path),
        "collision_guards_changed": False,
    }


@contextmanager
def cleaned_manual_model():
    """Scope cleanup to new manual contact/transport models, not frozen assets."""
    import layered_mound_checkpoint_runtime as runtime
    original = runtime._build_provisional_model

    def build(*args, **kwargs):
        manifest = original(*args, **kwargs)
        path = kwargs["output"] if "output" in kwargs else args[3]
        cleanup = clean_generated_model(path)
        return {**manifest,
                "model_sha256": cleanup["generated_model_after_sha256"],
                "legacy_push_shelf_cleanup": cleanup}

    runtime._build_provisional_model = build
    try:
        yield
    finally:
        runtime._build_provisional_model = original
