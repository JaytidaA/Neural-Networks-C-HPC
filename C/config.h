#ifndef NN_CONFIG_H
#define NN_CONFIG_H

#define ACTIVATION(X) sigmoid(X)
#define ACTIVATION_DERIV(X) sigmoid_deriv(X)

const float RSEED = 42.0f;

#define MAX_LINE_LEN 2048
#define MAX_COLS     128

#endif