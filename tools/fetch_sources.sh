#!/bin/sh
# Third-party code and input files used by zenke.py / maes.py (not stored in this repository). Run from the repository root.
set -e
Z=data/zenke2015_code; mkdir -p $Z/auryn data/maes2020_code
R=https://raw.githubusercontent.com/fzenke/pub2015orchestrated/master
for f in README.md globalvars.sh 1run_init.sh 2run_learn.sh 3run_cued.sh run_orchestrated_stdp.sh; do curl -sfL -o $Z/$f $R/$f; done
for f in sim_rc_p10c.cpp sim_rc_p11.cpp P10Connection.cpp P10Connection.h P11Connection.cpp P11Connection.h GlobalPFConnection.cpp GlobalPFConnection.h; do curl -sfL -o $Z/$f $R/src/$f; done
for f in shapes.pat shapes_cues.pat rf1.pat rf_discsR8.mtx; do curl -sfL -o $Z/$f $R/src/data/$f; done
A=https://raw.githubusercontent.com/fzenke/auryn/master/src/auryn
for f in AIF2Group.cpp AIF2Group.h AIFGroup.cpp AIFGroup.h IFGroup.cpp IFGroup.h StimulusGroup.cpp StimulusGroup.h STPConnection.cpp STPConnection.h NeuronGroup.cpp auryn_definitions.h System.cpp SpikingGroup.cpp SparseConnection.cpp Connection.cpp EulerTrace.cpp SpikeDelay.cpp DuplexConnection.cpp Trace.cpp; do curl -sfL -o $Z/auryn/$f $A/$f; done
M="https://raw.githubusercontent.com/ModelDBRepository/257609/master/Learning%20sequential%20dynamics"
for f in sim.jl runsim.jl; do curl -sfL -o data/maes2020_code/$f "$M/$f"; done
curl -sfL -o data/maes2020_code/readme.txt https://raw.githubusercontent.com/ModelDBRepository/257609/master/readme.txt
echo fetched
