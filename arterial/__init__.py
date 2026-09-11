#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import absolute_import

import sys

# The banner goes to stderr: stdout is reserved for program output, and the
# centerline branch extraction worker returns a pickle over stdout.
print("\nArterial framework for automated characterization of vascular tortuosity.", file=sys.stderr)
print("Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.", file=sys.stderr)
print("Code licensed under the PolyForm Noncommercial License 1.0.0. Noncommercial use only.\n", file=sys.stderr)
from . import *