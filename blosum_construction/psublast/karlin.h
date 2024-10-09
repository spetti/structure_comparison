#ifndef SIM_KARLIN_H
#define SIM_KARLIN_H
/* $Id: sim_karlin.h,v 1.1 1998/06/06 07:02:48 schwartz Exp $ */

int karlin(
	int low,		/* Lowest score (must be negative)    */
	int high,		/* Highest score (must be positive)   */
	double *pr,		/* Probabilities for various scores   */
	double *lambda,		/* Pointer to parameter lambda        */
	double *K		/* Pointer to parmeter K              */
);

#endif
