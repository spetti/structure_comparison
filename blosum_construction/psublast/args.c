#include "psublast/libc.h"
#include "psublast/types.h"
#include "psublast/misc.h"
#include "psublast/args.h"

#ifndef __lint
static const char rcsid[] =
"$Id: args.c,v 1.10 1999/10/01 18:47:52 schwartz Exp $";
#endif

static int argc;
static char **argv;
char *argv0;

/* ckargs  --  check that only certain parameters are set on the command line */
void ckargs(const char *options, int argcx, char **argvx, int non_options)
{
	int i;

	argc = argcx;
	argv = argvx;
	argv0 = argv0 ? argv0 : argv[0];
	for (i = non_options+1; i < argc; ++i)
		if (argv[i][1] != '=')
			fatalf("Improper command option: '%s'.", argv[i]);
		else if (!strchr(options, argv[i][0]))
			fatalf("Available options: %s\n", options);
}

/* get_argval  --------------------- get the value of a command-line argument */
bool get_argval(int c, int *val_ptr)
{
	int i;

	ck_argc("get_argval");
	for (i = 0; i < argc; ++i)
		if (argv[i][0] == c && argv[i][1] == '=') {
			*val_ptr = atoi(argv[i]+2);
			return 1;
		}
	return 0;
}

/* get_fargval  --------------- get the float value of a command-line argument */
bool get_fargval(int c, double *val_ptr)
{
        int i;

        ck_argc("get_fargval");
        for (i = 0; i < argc; ++i)
                if (argv[i][0] == c && argv[i][1] == '=') {
                        *val_ptr = atof(argv[i]+2);
                        return 1;
                }
        return 0;
}

bool get_cargval(int c, char **valp)
{
        int i;

        ck_argc("get_cargval");
        for (i = 0; i < argc; ++i)
                if (argv[i][0] == c && argv[i][1] == '=') {
                        *valp = argv[i]+2;
                        return 1;
                }
        return 0;
}

/* ck_argc - die if argc is unknown */
void ck_argc(const char *proc_name)
{
	if (argc == 0)
		fatalf("Call ckargs() before %s.\n", proc_name);
}

void fprintf_argv(FILE* fp)
{
	int i;
	fprintf(fp, "%s", argv0);
	for (i = 1; i < argc; ++i)
		(void)fprintf(fp, " %s", argv[i]);
}


void get_argval_min(int c, int *v, int d, int min, const char *msg)
{
	if (get_argval(c, v)) {
		if (*v < min)
			fatalf(msg, c);
	} else {
		*v = d;
	}
}	

void get_argval_max(int c, int *v, int d, int max, const char *msg)
{
	if (get_argval(c, v)) {
		if (*v > max)
			fatalf(msg, c);
	} else {
		*v = d;
	}
}

void get_argval_nonneg(int ch, int *val, int dflt)
{
	get_argval_min(ch, val, dflt, 0, "%c must be non-negative.");
}

void get_argval_pos(int ch, int *val, int dflt)
{
	get_argval_min(ch, val, dflt, 1, "%c must be positive.");
}

void argv_scores(argv_scores_t *s, const argv_scores_t *const dflt)
{
	*s = *dflt;

	if (get_argval('M', &s->M))
		if (s->M <= 0)
			fatal("M must be positive");

	if (get_argval('I', &s->I))
		if (s->I >= 0)
			fatal("I must be negative");

	if (get_argval('V', &s->V))
		if (s->V >= 0)
			fatal("V must be negative");

	if (s->V > s->I)
		fatal("transversions penalized less than transitions?");

	if (get_argval('O', &s->O))
		if (s->O <= 0)
			fatal("O must be positive");

	if (get_fargval('E', &s->E))
		if (s->E <= 0.0)
			fatal("E must be positive");

	if (get_fargval('S', &s->S))
		if (1.0 < s->S || s->S <= 0.0)
			fatal("S must be between 0.0 and 1.0");
}
