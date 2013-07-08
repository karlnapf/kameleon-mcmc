"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

echo collecting list of to be stopped jobs
qstat | grep ucabhst | awk '{printf "qdel %s\n",$1}' > temp.sh

echo stopping jobs
sh temp.sh

echo cleaning up
rm temp.sh
