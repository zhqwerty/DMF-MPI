#ifndef _EXAMPLES_H__
#define _EXAMPLES_H__
#include "../Tools/global_macros.h"

class Example {
public:
  int row, col; 
  double rating;
};
// typedef struct Example Example;

void write_binary_examples(struct Example* ex, int nExamples, const char *szFile);

Example* load_binary_examples(const char *szExample, int &nRows, int &nCols, int &nExamples);

Example* load_examples(const char *szExample, int &nRows, int &nCols, int &nExamples);


#endif
