#ifndef SIM_ARGS_H
#define SIM_ARGS_H
/* $Id: sim_args.h,v 1.7 1998/12/13 18:16:36 schwartz Exp $ */

typedef struct argv_scores {
	double E;
	int I;
	int M;
	int O;
	double S;
	int V;
} argv_scores_t;

bool get_argval(int, int *);
bool get_fargval(int, double *);
bool get_cargval(int, char **);
void ckargs(const char *, int , char **, int );
void fprintf_argv(FILE* );
void ck_argc(const char *);
void argv_scores(argv_scores_t*, const argv_scores_t *const);

void get_argval_min(int c, int *v, int d, int min, const char *msg);
void get_argval_max(int c, int *v, int d, int max, const char *msg);
void get_argval_nonneg(int ch, int *val, int dflt);
void get_argval_pos(int ch, int *val, int dflt);

extern char *argv0;
#endif
