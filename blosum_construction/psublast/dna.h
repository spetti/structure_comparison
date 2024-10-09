#ifndef SIM_DNA_H
#define SIM_DNA_H
/* $Id: sim_dna.h,v 1.4 1998/10/13 03:07:49 schwartz Exp $ */

#define DEFAULT_E 2
#define DEFAULT_I -10
#define DEFAULT_M 10
#define DEFAULT_O 60
#define DEFAULT_S 0.95
#define DEFAULT_V -10

void DNA_scores(argv_scores_t *ds, ss_t ss);
void DNA_scores_dflt(argv_scores_t *ds, ss_t ss, const argv_scores_t *dflt);
int DNA_thresh(SEQ *seq1, SEQ *seq2, argv_scores_t *ds);
int is_dchar(int ch);

#endif
