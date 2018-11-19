"""Line Integral Convolution algorithm and plotting tools.

:Author: Anne Archibald <peridot.faceted@gmail.com>
:Copyright: Copyright 2008 Anne Archibald.
:License: BSD-style license. See LICENSE.txt in the scipy source directory.
"""

import pyximport; pyximport.install()
from src.vectorplot.lic import *
from src.vectorplot.lic_internal import *
from src.vectorplot.kernels import *
