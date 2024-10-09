#ifndef SIM_BLAST_H
#define SIM_BLAST_H
/* $Id: blast.h,v 1.20 2000/03/15 21:27:23 schwartz Exp $ */

typedef uint32_t blast_ecode_t;

enum { BLAST_POS_TAB_SIZE = 65536UL };  /* 65536 = 4**8 */

typedef struct blast_table {
	int W;
	int num;
	const signed char *encoding;
	union {
		struct blast_pos_node **pos_tab;
		struct blast_epos_node **epos_tab;
	} u;
} blast_table_t;

#ifndef SCORE_T
#define SCORE_T long
#endif
typedef SCORE_T score_t;

typedef struct {
	int len, pos1, pos2;
	score_t score, cum_score;
	int filter;
} msp_t;

typedef struct msp_table {
	int size;
	int num;
	msp_t *msp;
} msp_table_t;
#define MSP_TAB_FIRST(t) ((t)->msp)
#define MSP_TAB_NEXT(m) ((m)+1)
#define MSP_TAB_NUM(t) ((t)->num)
#define MSP_TAB_MORE(t,m) ((m-MSP_TAB_FIRST(t))<MSP_TAB_NUM(t))
#define MSP_TAB_NTH(t,n) ((t)->msp[n])

typedef int (*msp_cmp_t)(const void *, const void *);

msp_table_t *msp_new_table(void);
blast_table_t *blast_table_new(SEQ *, int);
blast_table_t *blast_table_enc_new(SEQ *, int, const signed char *);
blast_table_t *blast_table_unmasked_new(SEQ *, int);
blast_table_t *blast_table_masked_new(SEQ *, int);
msp_table_t *blast_search(SEQ *seq1, SEQ *seq2, blast_table_t *bt, ss_t ss, int X, int K);
msp_t *msp_cons(int len, int pos1, int pos2, score_t score, int filter);
int msp_add(msp_table_t *mt, int len, int pos1, int pos2, score_t score, int filter);
int msp_extend_hit(msp_table_t *mt, SEQ *s1, SEQ *s2, ss_t ss, int X, int K, int W, int pos1, int pos2, int *diag_lev);
void blast_table_free(blast_table_t *bt);
void msp_free_table(msp_table_t *mt);

msp_table_t *msp_compress(msp_table_t *mt);
msp_table_t *msp_sort_by(msp_table_t *mt, msp_cmp_t cmp);
msp_table_t *msp_sort_pos1(msp_table_t *mt);
msp_table_t *msp_sort_pos2(msp_table_t *mt);
msp_table_t *msp_filter(msp_table_t *mt);
msp_table_t *msp_filter_old(msp_table_t *mt);
msp_table_t *msp_sort(msp_table_t *mt);
int msp_cmp_score(const void *e, const void *f);
int msp_cmp_pos1(const void *e, const void *f);
int msp_cmp_pos2(const void *e, const void *f);
void msp_print(SEQ *sf1, SEQ *sf2, msp_table_t *mt);
void msp_write(msp_table_t *mt);

typedef int (*connect_t)(msp_t *, msp_t *, int);
int msp_make_chain(msp_table_t *, int, int, int, connect_t);
#endif
