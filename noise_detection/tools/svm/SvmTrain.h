/*
 * SvmTrain.h
 *
 *  Created on: Sep 30, 2015
 *      Author: ray
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


class SvmTrain
{
private:
    struct svm_parameter param;		// set by parse_command_line
    struct svm_problem prob;		// set by read_problem
    struct svm_model *train_model;
    struct svm_node *x_space;
    int cross_validation;
    int nr_fold;
    char *train_line;
    int train_max_line_len;

private:
    void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
    void read_problem(const char *filename);
    void do_cross_validation();
    static void train_print_null(const char *s);
    void train_exit_with_help();
    void train_exit_input_error(int line_num);
    char* train_readline(FILE *input);
    int train_main(int argc, char **argv);

public:
    SvmTrain();
    void train(const char* _training);

};
