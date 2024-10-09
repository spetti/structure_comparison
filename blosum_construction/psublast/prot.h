#ifndef SIM_PROT_H
#define SIM_PROT_H
/* $Id: sim_prot.h,v 1.4 1998/10/29 21:48:13 schwartz Exp $ */

#define DEFAULT_AA_O 12
#define DEFAULT_AA_E 4

/* for protein substitution scores and the statistics-based threshold */
void protein_scores(argv_scores_t*, ss_t ss);
int protein_thresh(SEQ *seq1, SEQ *seq2, double S);
int is_pchar(int ch);

void protein_scores_pam60(argv_scores_t *ps, ss_t ss);
/*void protein_scores_gp(argv_scores_t *ps, ss_t ss, int smat[NP][NP]);*/
#endif
