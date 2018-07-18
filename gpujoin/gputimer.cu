#include "gputimer.hxx"

GPUTimer::EventPair* GPUTimer::add(const std::string & argName, cudaStream_t const& argStream) {

	EventPair* pair = new EventPair(argName, argStream);

	cudaEventCreate(&(pair->start));
	cudaEventCreate(&(pair->end));

	cudaEventRecord(pair->start, argStream);

	pairs.push_back(pair);
	return pair;
}

void GPUTimer::finish(EventPair* pair) {
    cudaEventRecord(pair->end, pair->stream);
    cudaEventSynchronize(pair->end);
}

float GPUTimer::sum(std::string const &argName) {
	float total = 0.0;
	std::vector<EventPair*>::iterator it = pairs.begin();
	for(; it != pairs.end(); ++it) {
		if ((*it)->name == argName) {
		    std::cout <<"hello there " << (*it)->name << std::endl;
			float millis = cudaEventElapsedTime(&millis,
										 (*it)->start,
										 (*it)->end);
			total += millis;
		}
	}
	return total;
}


GPUTimer::~GPUTimer() {
	std::vector<EventPair*>::iterator it = pairs.begin();
	for(; it != pairs.end(); ++it) {
		cudaEventDestroy((*it)->start);
		delete *it;
	}
}




std::ostream & operator<<(std::ostream & os, const GPUTimer & timer) {
	std::vector<GPUTimer::EventPair*>::const_iterator it = timer.pairs.begin();
	float millis;

	for(; it != timer.pairs.end(); ++it) {
		cudaEventElapsedTime(&millis,
							(*it)->start,
							(*it)->end);
		os << std::setw(16) << (*it)->name << std::setw(11) << millis << " ms" << std::endl;
	}
	return os;
}
