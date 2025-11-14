#ifndef NN_CONFIG_H
#define NN_CONFIG_H

#include <stdlib.h>
#include <time.h>

#define ACTIVATION(X) relu(X)
#define ACTIVATION_DERIV(X) relu_deriv(X)

static float RSEED = 46.0f;

#define MAX_LINE_LEN 2048
#define MAX_COLS     128

#endif
