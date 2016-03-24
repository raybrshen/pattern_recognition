/*
 * SvmScale.h
 *
 *  Created on: Sep 30, 2015
 *      Author: ray
 */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <unistd.h>

#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))


class SvmScale
{
private:
    char *scale_line;
    int scale_max_line_len;
    double lower,upper,y_lower,y_upper;
    int y_scaling;
    double *feature_max;
    double *feature_min;
    double y_max;
    double y_min;
    int max_index;
    int min_index;
    long int num_nonzeros;
    long int new_num_nonzeros;

private:
    void output_target(double value);
    void output(int index, double value);
    char* scale_readline(FILE *input);
    int clean_up(FILE *fp_restore, FILE *fp, const char *msg);
    void scale_exit_with_help();

public:
    SvmScale();
    int scale_main(int argc,char **argv);
    void scale_save(const char* _training);
    void scale_restore(const char* _testing, const char* _training);

};
