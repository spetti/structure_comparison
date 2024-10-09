#ifndef BLAST_ALIGN_H
#define BLAST_ALIGN_H

/* $Id: align.h,v 1.2 1999/04/16 20:04:34 schwartz Exp $ */

int align_basic(
        const uchar *s1, int len1,  
        const uchar *s2, int len2,
        int reverse, int xdrop_threshold, int match_cost, int mismatch_cost,
        int *e1, int *e2, edit_script_t *S);

#endif
