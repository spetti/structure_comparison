#ifndef SIM_EDIT_H
#define SIM_EDIT_H

/* $Id: edit.h,v 1.11 1999/04/16 20:04:34 schwartz Exp $ */

typedef unsigned int edit_op_t; /* 32 bits */

typedef struct {
	edit_op_t *op;			/* array of edit operations */
	unsigned int size, num;		/* size of allocation, number in use */
	edit_op_t last;			/* most recent operation added */
} edit_script_t;

#define EDIT_SCRIPT_TOP(es) (&((es)->op[(es)->num-1]))

enum {
	EDIT_OP_MASK = 0x3,
	EDIT_OP_ERR = 0x0,
	EDIT_OP_INS = 0x1,
	EDIT_OP_DEL = 0x2,
	EDIT_OP_REP = 0x3
};

edit_op_t *ed_ops_realloc(edit_op_t *op, unsigned int n);
edit_op_t *edit_script_first(edit_script_t *es);
edit_op_t *edit_script_next(edit_script_t *es, edit_op_t *op);
edit_op_t edit_op_cons(unsigned int op, unsigned int val);
edit_op_t edit_op_inc(edit_op_t op, unsigned int n);
edit_op_t edit_op_inc_last(edit_script_t *es, unsigned int n);
edit_script_t *edit_script_append(edit_script_t *es, edit_script_t *et);
edit_script_t *edit_script_concat(edit_script_t *es, edit_script_t *et);
edit_script_t *edit_script_copy(edit_script_t *);
edit_script_t *edit_script_fin(edit_script_t *data);
edit_script_t *edit_script_free(edit_script_t *es);
edit_script_t *edit_script_init(edit_script_t *es);
edit_script_t *edit_script_new(void);
edit_script_t *edit_script_reverse_inplace(edit_script_t *es);
edit_script_t *edit_script_transpose_inplace(edit_script_t *es);
int edit_script_del(edit_script_t *data, unsigned int k);
int edit_script_ins(edit_script_t *data, unsigned int k);
int edit_script_rep(edit_script_t *data, unsigned int k);
int edit_script_more(edit_script_t *data, unsigned int op, unsigned int k);
int edit_script_put(edit_script_t *es, unsigned int op, unsigned int n);
int edit_script_ready(edit_script_t *es, unsigned int n);
int edit_script_readyplus(edit_script_t *es, unsigned int n);
unsigned int edit_opc_get(edit_op_t op);
unsigned int edit_val_get(edit_op_t op);
void edit_script_prnt(edit_script_t *es);
int es_rep_len(edit_script_t *S, int *n, const uchar *p, const uchar *q, int *match);
int es_indel_len(edit_script_t *S, int *n, int *i, int *j);

#endif
