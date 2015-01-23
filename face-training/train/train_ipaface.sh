#!/usr/bin/env sh
CAFFEDIR=/home/jahausw/projects/caffe

$CAFFEDIR/build/tools/caffe train --solver=deepface_solver.prototxt
