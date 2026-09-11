#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Documentation consistency: code examples import real names, and links point
at files that exist.
"""

import ast
import glob
import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKDOWN = [os.path.join(REPO_ROOT, "README.md"), os.path.join(REPO_ROOT, "documentation", "installation.md"),
            os.path.join(REPO_ROOT, "tests", "README.md")] + glob.glob(os.path.join(REPO_ROOT, "arterial", "*", "README.md"))


def python_blocks(path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def module_top_level_names(module):
    """
    Returns the names defined at the top level of an arterial module, without importing it.

    """
    parts = module.split(".")
    base = os.path.join(REPO_ROOT, *parts)
    path = base + ".py" if os.path.isfile(base + ".py") else os.path.join(base, "__init__.py")
    if not os.path.isfile(path):
        return None
    names = set()
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


class TestCodeExamples(unittest.TestCase):

    def test_python_examples_parse(self):
        for path in MARKDOWN:
            for idx, block in enumerate(python_blocks(path)):
                with self.subTest(file=os.path.relpath(path, REPO_ROOT), block=idx):
                    try:
                        ast.parse(block)
                    except SyntaxError as error:
                        self.fail(f"example does not parse: {error}")

    def test_arterial_imports_in_examples_name_real_objects(self):
        for path in MARKDOWN:
            for block in python_blocks(path):
                try:
                    tree = ast.parse(block)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("arterial"):
                        with self.subTest(file=os.path.relpath(path, REPO_ROOT), module=node.module):
                            names = module_top_level_names(node.module)
                            self.assertIsNotNone(names, f"module {node.module} does not exist")
                            for alias in node.names:
                                if alias.name != "*":
                                    self.assertIn(alias.name, names, f"{node.module} does not define {alias.name}")


class TestLinks(unittest.TestCase):

    def test_relative_links_point_at_existing_files(self):
        for path in MARKDOWN:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            for target in re.findall(r"\]\(([^)#]+?)(?:#[^)]*)?\)", text):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                with self.subTest(file=os.path.relpath(path, REPO_ROOT), link=target):
                    resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
                    self.assertTrue(os.path.exists(resolved), f"broken link {target}")


class TestOutputPathsMatchCode(unittest.TestCase):

    def test_documented_output_files_are_the_ones_the_code_writes(self):
        with open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8") as handle:
            readme = handle.read()
        self.assertNotIn("_segmentation.nii.gz", readme)
        self.assertNotIn("landmarks/", readme)
        for name in ["segmentation.nii.gz", "landmarks.json", "landmarks_slicer.json", "branch_model.vtk", "centerline_segments_array.npy",
                     "segments_graph_pred.pickle", "local_graph.pickle"]:
            self.assertIn(name, readme)
