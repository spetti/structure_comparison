#ifndef SIM_PRNT_H
#define SIM_PRNT_H
/* $Id: prnt.h,v 1.10 1999/09/23 21:09:52 schwartz Exp $ */

void print_align_header(SEQ *seq1, SEQ *seq2, argv_scores_t *ds);
void print_align_header_n(SEQ *seq1, SEQ *seq2, argv_scores_t *ds, int n);

void print_align(int score, uchar *seq1, uchar *seq2, int beg1, int end1, int beg2, int end2, int *script);
void print_block(int score, int beg1, int end1, int beg2, int end2, int f, int pm);
void print_align_lav(int score, const uchar *seq1, const uchar *seq2, int beg1, int end1, int beg2, int end2, edit_script_t *script);
void print_align_summary(int score, const uchar *seq1, const uchar *seq2, int beg1, int end1
, int beg2, int end2, edit_script_t *script);
typedef struct align_ {
        int beg1, beg2, end1, end2;
        struct align_ *next_align;
        edit_script_t *script;
	int score;
	uchar *seq1, *seq2;
} align_t;

void print_align_list(align_t *a);
void free_align_list(align_t *a);
int align_match_percent(int run, int match);

#endif
