from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))
import legacy_push_shelf_cleanup as cleanup

FIXTURE = ('<geom name="r1_ground_push_shelf" type="box" pos="5.13 .65 2.86" '
           'size=".6 .55 .03" quat="1 0 0 0" contype="1024" conaffinity="256"/>')
XML = ('<mujoco><worldbody><geom name="ground_plane" type="plane"/>' + FIXTURE +
       '<geom name="r1_ground_real_ramp" type="box"/>'
       '<geom name="r1_truck_floor" type="box"/>'
       '<body name="r1-rock-000"><freejoint/><geom name="rock"/></body>'
       '</worldbody><actuator><motor name="bucket"/></actuator></mujoco>')


class ShelfCleanupTest(unittest.TestCase):
    def test_only_shelf_removed_and_second_call_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"model.xml";p.write_text(XML)
            result=cleanup.clean_generated_model(p)
            expected=ET.fromstring(XML);world=expected.find('worldbody')
            world.remove(world.find("geom[@name='r1_ground_push_shelf']"))
            actual=ET.parse(p).getroot()
            def topology(node):
                return node.tag, node.attrib, [topology(c) for c in node]
            self.assertEqual(topology(actual),topology(expected))
            self.assertEqual(result['removed_count'],1)
            self.assertFalse(result['collision_guards_changed'])
            before=p.read_bytes();again=cleanup.clean_generated_model(p)
            self.assertEqual(again['removed_count'],0)
            self.assertEqual(p.read_bytes(),before)

    def test_changed_duplicate_nested_or_referenced_fixture_is_not_deleted(self):
        cases=[XML.replace('5.13 .65 2.86','5.13 .65 0'),
               XML.replace(FIXTURE,FIXTURE*2),
               XML.replace(FIXTURE,'<body>'+FIXTURE+'</body>'),
               XML.replace('</mujoco>','<contact><pair geom1="r1_ground_push_shelf" geom2="rock"/></contact></mujoco>')]
        for xml in cases:
            with self.subTest(xml=xml),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/"model.xml";p.write_text(xml)
                with self.assertRaises(ValueError):cleanup.clean_generated_model(p)
                self.assertEqual(p.read_text(),xml)

    def test_context_updates_identity_and_restores_builder_on_failure(self):
        import layered_mound_checkpoint_runtime as runtime
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"model.xml"
            def build(*args,**kwargs):
                p.write_text(XML)
                return {'model_sha256':cleanup.sha256(p),'source':'unchanged'}
            with patch.object(runtime,'_build_provisional_model',side_effect=build) as original:
                with self.assertRaisesRegex(RuntimeError,'deliberate'):
                    with cleanup.cleaned_manual_model():
                        manifest=runtime._build_provisional_model(None,None,None,p)
                        self.assertEqual(manifest['model_sha256'],cleanup.sha256(p))
                        self.assertEqual(manifest['source'],'unchanged')
                        self.assertEqual(manifest['legacy_push_shelf_cleanup']['removed_count'],1)
                        raise RuntimeError('deliberate')
                self.assertIs(runtime._build_provisional_model,original)


if __name__=='__main__':unittest.main()
