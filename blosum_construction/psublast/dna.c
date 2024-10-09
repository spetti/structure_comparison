#include "psublast/libc.h"
#include "psublast/types.h"
#include "psublast/seq.h"
#include "psublast/misc.h"
#include "psublast/args.h"
#include "psublast/karlin.h"
#include "psublast/dna.h"

#ifndef __lint
static const char rcsid[] =
"$Id: dna.c,v 1.12 1999/10/01 22:08:49 schwartz Exp $";
#endif

static const argv_scores_t EIMOSV = { 
	DEFAULT_E, 
	DEFAULT_I,
	DEFAULT_M,
	DEFAULT_O,
	DEFAULT_S,
	DEFAULT_V
};

/* DNA characters */
const uchar dchars[] = "ABCDGHKMNRSTVWXY";

/* DNA_scores -----------------------------------  set scoring matrix for DNA */
void DNA_scores_dflt(argv_scores_t *ds, ss_t ss, const argv_scores_t *dflt)
{
	int i, j, bad;

	ck_argc("DNA_scores");
	argv_scores(ds, dflt);

	for (i = 0; i < NACHARS; ++i)
		for (j = 0; j < NACHARS; ++j)
			ss[i][j] = ds->V;

	bad = -100*ds->M;
	for (i = 0; i < NACHARS; ++i)
		ss['X'][i] = ss[i]['X'] = bad;

	ss['a']['a'] = ss['c']['c'] = ss['g']['g'] = ss['t']['t'] = ds->M;
	ss['a']['A'] = ss['c']['C'] = ss['g']['G'] = ss['t']['T'] = ds->M;
	ss['A']['a'] = ss['C']['c'] = ss['G']['g'] = ss['T']['t'] = ds->M;
	ss['A']['A'] = ss['C']['C'] = ss['G']['G'] = ss['T']['T'] = ds->M;

	ss['a']['g'] = ss['g']['a'] = ss['c']['t'] = ss['t']['c'] = ds->I;
	ss['a']['G'] = ss['g']['A'] = ss['c']['T'] = ss['t']['C'] = ds->I;
	ss['A']['g'] = ss['G']['a'] = ss['C']['t'] = ss['T']['c'] = ds->I;
	ss['A']['G'] = ss['G']['A'] = ss['C']['T'] = ss['T']['C'] = ds->I;
}

void DNA_scores(argv_scores_t *ds, ss_t ss)
{
	DNA_scores_dflt(ds, ss, &EIMOSV);
}

static int count_acgt(SEQ *s, int *A, int *C, int *G, int *T)
{
	int i;
	int len = SEQ_LEN(s);

        for (i = 0; i < len; ++i) {
                int c = SEQ_AT(s,i);
		switch (c) {
		case 'A': case 'a':
			++*A;
			break;
		case 'C': case 'c':
			++*C;
			break;
		case 'G': case 'g':
			++*G;
			break;
		case 'T': case 't':
			++*T;
			break;
		}
	}
	return *A + *C + *G + *T;
}

/* DNA_thresh ----------- determine cutoff score according to Karlin-Altschul */
int DNA_thresh(SEQ *seq1, SEQ *seq2, argv_scores_t *ds)
{
        double *pvec, *p, pMatch, pTransit, pTransver;
	double N, param_K, param_lambda, x;
        int A1, A2, C1, C2, G1, G2, T1, T2;
        int i, len1, len2, plen;
	int Match = ds->M, Transver = ds->V, Transit = ds->I;

        A1 = A2 = C1 = C2 = G1 = G2 = T1 = T2 = 0;
	len1 = count_acgt(seq1, &A1, &C1, &G1, &T1);
	len2 = count_acgt(seq2, &A2, &C2, &G2, &T2);

        pMatch = ((double)A1)*((double)A2)
               + ((double)C1)*((double)C2)
               + ((double)G1)*((double)G2)
               + ((double)T1)*((double)T2);
        pTransit = ((double)A1)*((double)G2)
               + ((double)C1)*((double)T2)
               + ((double)G1)*((double)A2)
               + ((double)T1)*((double)C2);
	if (seq1 == seq2) {
	    N = ((double)len1-1)*((double)len1)/2.0;	
	    pMatch = (pMatch-(double) len1)/2.0;
	    pTransit /= 2.0;
	} else {
	    N = ((double)len1)*((double)len2);	    
	}
        pTransver = N - pMatch - pTransit;

	plen = -Transver+Match;
	assert(plen > 0);

	pvec = ckalloc(sizeof(pvec[0]) * (plen+1));
        for (i = 0; i <= plen; ++i)
                pvec[i] = 0.0;
        p = pvec - Transver;
        p[Transver] = pTransver/N;
        p[Transit] += (pTransit/N);
        p[Match] += (pMatch/N);

        if (karlin(Transver, Match, pvec, &param_lambda, &param_K) == 0)
                fatal("compution of significance levels failed");
	ZFREE(pvec);

        x = log(ds->S)/(-param_K*N);
        return (int)(2*log(sqrt(x))/(-param_lambda));
}

int is_dchar(int ch)
{
	return !!strchr((const char*)dchars, toupper(ch));
}

