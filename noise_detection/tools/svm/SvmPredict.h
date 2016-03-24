/*
 * SvmPredict.h
 *
 *  Created on: Sep 30, 2015
 *      Author: ray
 */

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "svm.h"

static int (*predict_info)(const char *fmt,...) = &printf;

class SvmPredict
{
private:
    struct svm_node *x;
    int max_nr_attr;
    struct svm_model* predict_model;
    int predict_probability;
    char *predict_line;
    int predict_max_line_len;

public:
    double last_predicted_label;

private:
    static int predict_print_null(const char *s,...);
    char* predict_readline(FILE *input);
    void predict_exit_input_error(int line_num);
    void sub_predict(FILE *input, FILE *output);
    void predict_exit_with_help();
    int predict_main(int argc, char **argv);

public:
    SvmPredict();
    void predict(const char* _testing, const char* _training);
};
