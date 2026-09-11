#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Packaging consistency: setup.py must declare what the package imports and
ship every package directory.
"""

import ast
import glob
import os
import sys
import unittest

from setuptools import find_packages

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(REPO_ROOT, "arterial")

# import name -> distribution name in setup.py
IMPORT_TO_DISTRIBUTION = {
    "cc3d": "connected-components-3d",
    "skimage": "scikit-image",
    "ants": "antspyx",
    "SimpleITK": "SimpleITK",
    "torch_geometric": "torch_geometric",
}
CONDA_ONLY = {"vmtk"}


def _setup_metadata():
    """
    Extracts install_requires, extras_require and package_data from setup.py
    without executing it.

    """
    tree = ast.parse(open(os.path.join(REPO_ROOT, "setup.py"), encoding="utf-8").read())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "setup")
    kwargs = {k.arg: k.value for k in call.keywords}
    literal = lambda key: ast.literal_eval(kwargs[key]) if key in kwargs else None
    return literal("install_requires") or [], literal("extras_require") or {}, literal("package_data") or {}


def _third_party_imports():
    """
    Collects top-level third-party module names imported anywhere in the package.

    """
    stdlib = set(sys.stdlib_module_names)
    found = {}
    for path in glob.glob(os.path.join(PACKAGE_DIR, "**", "*.py"), recursive=True):
        if os.sep + "models" + os.sep in path:
            continue
        for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top not in stdlib and top != "arterial":
                    found.setdefault(top, set()).add(os.path.relpath(path, REPO_ROOT))
    return found


class TestPackaging(unittest.TestCase):

    def setUp(self):
        self.install_requires, self.extras_require, self.package_data = _setup_metadata()
        declared = {req.split("[")[0].split("=")[0].split("<")[0].split(">")[0].strip() for req in self.install_requires}
        for reqs in self.extras_require.values():
            declared |= {req.split("=")[0].strip() for req in reqs}
        self.declared = {name.lower() for name in declared}

    def test_every_import_is_a_declared_dependency(self):
        for module, files in sorted(_third_party_imports().items()):
            if module in CONDA_ONLY:
                continue
            distribution = IMPORT_TO_DISTRIBUTION.get(module, module)
            with self.subTest(module=module):
                self.assertIn(distribution.lower(), self.declared, f"{module} imported by {sorted(files)} but not declared in setup.py")

    def test_no_stale_package_data(self):
        for package, patterns in self.package_data.items():
            base = os.path.join(REPO_ROOT, *package.split("."))
            for pattern in patterns:
                with self.subTest(pattern=f"{package}:{pattern}"):
                    self.assertTrue(glob.glob(os.path.join(base, pattern)), "package_data pattern matches nothing")

    def test_every_source_directory_is_a_package(self):
        packages = set(find_packages(where=REPO_ROOT, exclude=["tests", "tests.*"]))
        for path in glob.glob(os.path.join(PACKAGE_DIR, "**", "*.py"), recursive=True):
            rel = os.path.relpath(os.path.dirname(path), REPO_ROOT).replace(os.sep, ".")
            if ".models" in rel or rel.endswith("models"):
                continue
            with self.subTest(directory=rel):
                self.assertIn(rel, packages, f"{rel} has .py files but no __init__.py, so it is not installed")

    def test_optional_dependencies_are_not_imported_at_module_level(self):
        optional = {"SimpleITK", "dicom2nifti", "ants"}
        for path in [os.path.join(PACKAGE_DIR, "io", "dicom_and_nifti.py"), os.path.join(PACKAGE_DIR, "io", "registration.py")]:
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in tree.body:
                names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else \
                        [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module else []
                for name in names:
                    self.assertNotIn(name, optional, f"{os.path.basename(path)} imports optional {name} at module level")

    def test_io_modules_import_without_optional_dependencies(self):
        import arterial.io.dicom_and_nifti  # noqa: F401
        import arterial.io.registration  # noqa: F401
