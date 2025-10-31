#ifndef NN_CONFIG_H
#define NN_CONFIG_H

#define ACTIVATION(X) relu(X)
#define ACTIVATION_DERIV(X) relu_deriv(X)

const float RSEED = 42.0f;

#define MAX_LINE_LEN 2048
#define MAX_COLS     128

#endif