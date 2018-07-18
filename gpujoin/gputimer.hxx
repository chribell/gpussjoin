#ifndef GPU_JOIN_TIMER_HXX
#define GPU_JOIN_TIMER_HXX

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>


class GPUTimer {
	public:
		struct EventPair {
			std::string name;
			cudaEvent_t start;
			cudaEvent_t end;
			cudaStream_t stream;
			EventPair(std::string const& argName, cudaStream_t const& argStream) : name(argName), stream(argStream)  {}
		};
	protected:
		std::vector<EventPair*> pairs;
	public:
		EventPair* add(std::string const& argName, cudaStream_t const& argStream);
		float sum(std::string const& argName);
		void finish(EventPair* pair);
		~GPUTimer();
		friend std::ostream & operator<<(std::ostream & os, const GPUTimer & timer);
};

#endif //GPU_JOIN_TIMER_HXX
