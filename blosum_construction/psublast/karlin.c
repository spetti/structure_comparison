#include "psublast/libc.h"
#include "psublast/types.h"
#include "psublast/misc.h"
#include "psublast/karlin.h"

#ifndef __lint
static const char rcsid[] =
"$Id: karlin.c,v 1.6 1999/10/01 18:47:52 schwartz Exp $";
#endif

static int gcd(int a, int b)
{
        int c;

        if (b<0) b= -b;
        if (b>a) { c=a; a=b; b=c; }
        while (b) { 
            a = a % b; 
            if (!a) return b;
            b = b % a;
        }
        return a;
}
      
/**************** Statistical Significance Parameter Subroutine ****************
        Version 1.0     February 2, 1990
        Version 1.1     July 5,     1990

	The current version was slighly modified by Zheng Zhang, Aug. 1996

        Original Program by:     Stephen Altschul

        Address:        National Center for Biotechnology Information
                        National Library of Medicine
                        National Institutes of Health
                        Bethesda, MD  20894

        Internet:       altschul@ncbi.nlm.nih.gov

See:    Karlin, S. & Altschul, S.F. "Methods for Assessing the Statistical
        Significance of Molecular Sequence Features by Using General Scoring
        Schemes,"  Proc. Natl. Acad. Sci. USA 87 (1990), 2264-2268.

        Computes the parameters lambda and K for use in calculating the
        statistical significance of high-scoring segments or subalignments.

        The scoring scheme must be integer valued.  A positive score must be
        possible, but the expected (mean) score must be negative.

        A program that calls this routine must provide the value of the lowest
        possible score, the value of the greatest possible score, and a pointer
        to an array of probabilities for the occurence of all scores between
        these two extreme scores.  For example, if score -2 occurs with
        probability 0.7, score 0 occurs with probability 0.1, and score 3
        occurs with probability 0.2, then the subroutine must be called with
        low = -2, high = 3, and pr pointing to the array of values
        { 0.7, 0.0, 0.1, 0.0, 0.0, 0.2 }.  The calling program must also
	provide pointers to lambda and K; the subroutine will then calculate
	the values of these two parameters.  In this example, lambda = 0.330
	and K=0.154.

        The parameters lambda and K can be used as follows.  Suppose we are
        given a length N random sequence of independent letters.  Associated
        with each letter is a score, and the probabilities of the letters
        determine the probability for each score.  Let S be the aggregate score
        of the highest scoring contiguous segment of this sequence.  Then if N
        is sufficiently large (greater than 100), the following bound on the
        probability that S is greater than or equal to x applies:
        
                P( S >= x )   <=   1 - exp [ - KN exp ( - lambda * x ) ].
        
        In other words, the p-value for this segment can be written as
        1-exp[-KN*exp(-lambda*S)].

        This formula can be applied to pairwise sequence comparison by
	assigning scores to pairs of letters (e.g. amino acids), and by
	replacing N in the formula with N*M, where N and M are the lengths
	of the two sequences being compared.

        In addition, letting y = KN*exp(-lambda*S), the p-value for finding m
        distinct segments all with score >= S is given by:

                               2             m-1           -y
                1 - [ 1 + y + y /2! + ... + y   /(m-1)! ] e

        Notice that for m=1 this formula reduces to 1-exp(-y), which is the
	same as the previous formula.

*******************************************************************************/
#define MAXIT 150       /* Maximum number of iterations used in calculating K */
int karlin(
	int low,		/* Lowest score (must be negative)    */
	int high,		/* Highest score (must be positive)   */
	double *pr,		/* Probabilities for various scores   */
	double *lambda,		/* Pointer to parameter lambda        */
	double *K		/* Pointer to parmeter K              */    )
{
        int i,j,range,lo,hi,first;
        double up,new,Sum,av, tmp1, tmp2;
        register double sum;
        double *p,*P,*ptrP;
        register double *ptr1,*ptr2,*ptr1e;

        /* Check that scores and their associated probabilities are valid     */
        if (low>=0) 
            fatal("Lowest score must be negative.");
        for (i=range=high-low;i> -low && !pr[i];--i)
		/*LINTED empty loop body*/; 
        if (i<= -low) 
            fatal("A positive score must be possible.");
        for (sum=0.0,i=0; i <= range; sum += pr[i++]) 
            if (pr[i]<0) 
                fatal("Negative probabilities not allowed.");
        if (sum<0.99995 || sum>1.00005) 
            (void)fprintf(stderr,
                          "Probabilities sum to %.4f.  Normalizing.\n",sum);
        p = (double *) ckalloc((range+1)*sizeof(double));
        for (Sum = (double)low, i = 0; i<=range; ++i) 
            Sum += i*(p[i]=pr[i]/sum);
        if (Sum>=0) {
            (void)fprintf(stderr,
                          "Invalid (non-negative) expected score:  %.3f\n", Sum);
            exit(1);
	}

        /* Calculate the parameter lambda */

        up=0.5;
        do {
            up += up;
            ptr1 = p;
            tmp1 = exp(up); tmp2 = exp(up*(low-1));
            for (sum = 0, i = low; i <= high; ++i) 
                sum += (*ptr1++)*(tmp2 *= tmp1);
        } while (sum < 1.0);
        for (*lambda = 0.0, j = 0; j < 25; ++j) {
            new=(*lambda+up)/2.0;
            ptr1=p; tmp1 = exp(new); tmp2 = exp(new*(low-1));
            for (sum = 0, i = low;i <= high; ++i) 
                sum += (*ptr1++) * (tmp2 *= tmp1);
            if (sum > 1.0) up=new;
            else *lambda=new;
        }

        /* Calculate the pamameter K */

        ptr1=p; tmp1 = exp(*lambda);
        for (av=0,i=low;i<=high;++i) 
            av+= *ptr1++ *i* exp(*lambda * i);
        if (low == -1 || high == 1) {
                *K = (high == 1) ? av : Sum*Sum/av;
                *K *= 1.0 - 1.0/tmp1;
                free(p);
                return 1;       /* Parameters calculated successfully */
        }
        Sum=0.0; lo=hi=0;
        P= (double *) ckalloc(MAXIT*(range+2)*sizeof(double));
        for (*P=sum=1.0,j=1; j<=MAXIT && sum>0.00001; Sum+=sum/=j++) {
                first=range;
                for (ptrP=P+(hi+=high)-(lo+=low), ptr1e = ptrP-range; 
		     ptrP>=P; *ptrP-- =sum) {
                        ptr1=ptrP-first;
                        ptr2=p+first;
                        for (sum=0; ptr1 >= ptr1e; ptr1--, ptr2++)
                            if (*ptr1 != 0.0) {
                                sum += (*ptr1) * (*ptr2);
                            }
                        if (first) --first;
                        if (ptr1e > P) --ptr1e;
                }
                for (sum=0,i=lo;i;++i) 
                    sum += *++ptrP * exp(*lambda * i);
                for (;i<=hi;++i) sum+= *++ptrP;
        }
        if (j>MAXIT) 
	    (void)fprintf(stderr,
                          "Insufficient iterations in calculating param_K.\n");
        for (i=low;!p[i-low];++i) /*LINTED empty loop body*/; 
        for (j= -i;i<high && j>1;) 
            if (p[++i-low] != 0) j=gcd(j,i);
        *K = ((double) j*exp(-2.0*Sum))/(av*(1.0-exp(- *lambda*j)));

        free(p);
        free(P);
        return 1;               /* Parameters calculated successfully */
}

