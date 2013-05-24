echo collecting list of to be stopped jobs
qstat | grep ucabhst | awk '{printf "qdel %s\n",$1}' > temp.sh

echo stopping jobs
sh temp.sh

echo cleaning up
rm temp.sh
