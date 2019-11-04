#!/usr/bin/env bash

function build() {
    # $1 => folder name
    rm -rf $1
    mkdir $1
    cd $1
    #-DCMAKE_RULE_MESSAGES:BOOL=OFF -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
    CC=gcc-8 CXX=g++-8 cmake -DCMAKE_BUILD_TYPE=$1 CMAKE_CXX_FLAGS=-m64 -DSM_ARCH=$2 ..
    make -j 4
}

if [ $# -ne 2 ]; then
    echo "Argument (Release|Debug) needed"
    echo "Argument (SM arch, compute capability) needed"
    exit 1
fi

if [ "$1" = "Release" ]; then
    build Release $2
elif [ "$1" = "Debug" ]; then
    build Debug $2
else
    echo "Wrong build argument"
    exit 1
fi
