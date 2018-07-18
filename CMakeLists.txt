cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(gpussjoin CXX CUDA)

set(SM_ARCH 61 CACHE STRING "CUDA SM architecture e.g. 50, 52, 61")

set(NVCC_FLAGS --expt-extended-lambda -gencode arch=compute_${SM_ARCH},code=sm_${SM_ARCH})

find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
#cuda requirements
include_directories(/opt/cuda/include)

set(setsimjoinfiles main.cxx input.cxx statistics.cxx timing.cxx
		cmdline_allpairs.cxx
		cmdline_groupjoin.cxx
		cmdline_groupjoin_jaccard.cxx
		cmdline_mpjoin.cxx
		cmdline_mpjoin_jaccard.cxx
		)

add_executable(set_sim_join ${setsimjoinfiles})
set_property(TARGET set_sim_join PROPERTY CXX_STANDARD 11)
target_compile_options(set_sim_join PRIVATE -O3)

target_link_libraries( set_sim_join ${Boost_LIBRARIES}  )

add_library(gpujoin STATIC
		gpu_handler.cu
		gpujoin/gputimer.cu
        gpujoin/scenarios.cu
)

set_property(TARGET gpujoin PROPERTY CUDA_STANDARD 11)
target_compile_options(gpujoin PRIVATE ${NVCC_FLAGS})

target_link_libraries(set_sim_join gpujoin)
