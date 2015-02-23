#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define USE_WINDOWS_API 1
#include <stdint.h>

extern "C" bool __stdcall QueryPerformanceCounter(uint64_t *);
extern "C" bool __stdcall QueryPerformanceFrequency(uint64_t *);
#elif defined(_APPLE_) || defined(__APPLE__) || \
    defined(APPLE)   || defined(_APPLE)    || defined(__APPLE) || \
    defined(unix)    || defined(__unix__)  || defined(__unix)
#define USE_WINDOWS_API 0
#include <unistd.h>
#include <sys/time.h>
#endif

// Get current time (measured in milliseconds).
inline double currentTime()
{
#if USE_WINDOWS_API
	uint64_t t, freq;
	QueryPerformanceCounter(&t);
	QueryPerformanceFrequency(&freq);
	return (t * 1000.0) / freq;
#else
    struct timeval t;
    gettimeofday(&t, NULL);
    return (t.tv_usec/1000.0 + t.tv_sec*1000.0);
#endif
}
