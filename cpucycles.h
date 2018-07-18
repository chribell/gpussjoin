#ifndef CPUCYCLES_H
#define CPUCYCLES_H

/* Copyright 2014-2015 Willi Mann
 *
 * This file is part of set_sim_join.
 *
 * set_sim_join is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Foobar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with set_sim_join.  If not, see <http://www.gnu.org/licenses/>.
 */

typedef unsigned long long ticks;

static __inline__ ticks cpu_cycles_start (void) {
  unsigned cycles_low, cycles_high;
  asm volatile ("CPUID\n\t"
		"RDTSC\n\t"
		"mov %%edx, %0\n\t"
		"mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low)::
		"%rax", "%rbx", "%rcx", "%rdx");
  return ((ticks)cycles_high << 32) | cycles_low;
}

static __inline__ ticks cpu_cycles_stop (void) {
  unsigned cycles_low, cycles_high;
  asm volatile("RDTSCP\n\t"
	       "mov %%edx, %0\n\t"
	       "mov %%eax, %1\n\t"
	       "CPUID\n\t": "=r" (cycles_high), "=r" (cycles_low):: "%rax",
	       "%rbx", "%rcx", "%rdx");
  return ((ticks)cycles_high << 32) | cycles_low;
}

#endif
