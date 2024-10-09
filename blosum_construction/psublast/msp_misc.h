#ifndef MSP_MISC_H
#define MSP_MISC_H
/* $Id: msp_misc.h,v 1.1 1999/04/16 17:24:40 schwartz Exp $ */

#define MSP_CONTAINS(outer,inner) \
        ((outer)->b1 <= (inner)->b1 && (outer)->b2 <= (inner)->b2 && \
         (outer)->e1 >= (inner)->e1 && (outer)->e2 >= (inner)->e2)

#define MSP_CLOSE_ENOUGH 10 /* XXX - arbitrary */

#define MSP_CLOSE(a,b) \
        (abs(((a)->b1 - (a)->b2) - ((b)->b1 - (b)->b2)) < MSP_CLOSE_ENOUGH)

#endif
