#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <time.h>

// Important configuration
#include "config.h"

// Check pointer parameters
#define CHECK_PARAM(X) \
do { if (!(X)) { \
    fprintf(stderr, "%s: NULL parameter passed: %s\n", __func__, #X); \
    exit(EXIT_FAILURE); \
} } while (0)

// Call after malloc
#define MALLOC_CHECK(X) \
do { if (!(X)) { \
    fprintf(stderr, "%s: malloc failed for %s: %s\n", __func__, #X, strerror(errno)); \
    exit(EXIT_FAILURE); \
} } while (0)

#define MATINDEX(i, j, cols) (((i) * (cols)) + (j))

typedef struct {
    double value;
    double bias;
} neuron;

typedef struct {
    size_t nneurons;
    neuron *neurons;
} layer;

typedef double connection;

// Assume a classification model
typedef struct {
    // Include input and output as well
    size_t nlayers;
    layer *_layers;

    // Assume a fully connected layer, can set connections to 0.0 for dropout
    size_t *_nconnections;
    // _connections is an array of flat matrices stored in row-major order
    connection **_connections;
} model;

// Assume that final feature is the target feature which has the classes [0, n_classes) \cap \mathbb{Z}
typedef struct {
    size_t n_samples;
    size_t n_features;
    double **data;
} DataFrame;

// Important model functions
model init_model(size_t number_layers, size_t *neurons_per_layer);
DataFrame read_csv(const char *filename);
void train_model(model neural_network, const DataFrame df, double alpha, unsigned epochs, double threshold);
double *test_model(const model neural_network, const DataFrame df);
double accuracy_metric(DataFrame df, double *y);
void free_model(model neural_network);
void free_dataframe(DataFrame df);

// Important math functions
static inline double sigmoid(double x);
static inline double sigmoid_deriv(double x);
static inline double relu(double x);
static inline double relu_deriv(double x);
static inline void softmax(size_t nmemb, neuron *neurons);

int main(void)
{
    srand((unsigned) RSEED);

    size_t nlayers = 4;
    size_t npl[] = {13, 16, 16, 2};
    model NeuralNetwork = init_model(nlayers, npl);
    clock_t start, end;

    DataFrame df = read_csv("../datasets/heart_disease_train.csv");

    start = clock();
        train_model(NeuralNetwork, df, 0.01, 2000, 0.1);
    end = clock();
    puts("");

    printf(
        "Time taken to train model in pure C, highly unoptimised code: %lf\n",
        ((double) (end - start)) / CLOCKS_PER_SEC
    );

    free_dataframe(df);
    df  = read_csv("../datasets/heart_disease_test.csv");

    double *y = test_model(NeuralNetwork, df);
    double acc = accuracy_metric(df, y);
    printf("The accuracy of the trained model is: %lf%%\n", acc * 100.0);

    free(y);
    free_dataframe(df);
    free_model(NeuralNetwork);

    return 0;
}

model init_model(size_t nl, size_t *npl)
{
    CHECK_PARAM(npl);

    layer *_layers = (layer *) malloc(nl * sizeof(layer));
    MALLOC_CHECK(_layers);

    for (size_t i = 0; i < nl; i++) {
        _layers[i].nneurons = npl[i];
        _layers[i].neurons = (neuron *) calloc(sizeof(neuron), npl[i]);
        MALLOC_CHECK(_layers[i].neurons);
    }

    size_t *_nconnections = (size_t *) malloc((nl - 1) * sizeof(size_t));
    MALLOC_CHECK(_nconnections);

    connection **_connections = (connection **) malloc((nl - 1) * sizeof(connection *));
    MALLOC_CHECK(_connections);

    for (size_t i = 0; i < nl - 1; i++) {
        _nconnections[i] = npl[i] * npl[i + 1];
        _connections[i] = (connection *) malloc(sizeof(connection) * npl[i] * npl[i + 1]);
        MALLOC_CHECK(_connections[i]);
        for (size_t j = 0; j < npl[i] * npl[i + 1]; j++)
            _connections[i][j] = ((double) rand()) / RAND_MAX;
    }

    return (model) {
        .nlayers = nl,
        ._layers = _layers,
        ._nconnections = _nconnections,
        ._connections = _connections,
    };
}

// Imma be real with you, I asked ChatGPT to help me write this function
DataFrame read_csv(const char *filename)
{
    CHECK_PARAM(filename);

    FILE *fp = fopen(filename, "r");
    CHECK_PARAM(fp);

    char line[MAX_LINE_LEN];
    DataFrame df;

    // --- Read header line ---
    if (!fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "Empty file\n");
        fclose(fp);
        return (DataFrame) {0, 0, NULL};
    }

    // Count columns
    size_t n_features = 0;
    for (char *p = line; *p; ++p)
        if (*p == ',') n_features++;
    n_features++;  // last column

    df.n_features = n_features;

    // --- Count rows (n_samples) ---
    size_t n_samples = 0;
    while (fgets(line, sizeof(line), fp))
        if (line[0] != '\n' && line[0] != '\r')
            n_samples++;

    df.n_samples = n_samples;

    // --- Allocate data array ---
    df.data = malloc(n_samples * sizeof(double *));
    MALLOC_CHECK(df.data);

    for (size_t i = 0; i < n_samples; i++) {
        df.data[i] = malloc(n_features * sizeof(double));
        MALLOC_CHECK(df.data[i]);
    }

    // --- Read data again ---
    rewind(fp);
    fgets(line, sizeof(line), fp); // skip header

    char *token;
    size_t row = 0;
    while (fgets(line, sizeof(line), fp) && row < n_samples) {
        token = strtok(line, ",\n\r");
        size_t col = 0;
        while (token && col < n_features) {
            df.data[row][col] = atof(token);
            token = strtok(NULL, ",\n\r");
            col++;
        }
        row++;
    }

    fclose(fp);
    return df;
}

void train_model(model nn, const DataFrame df, double a, unsigned epochs, double th)
{
    // e = epochs
    // r = rows of df
    // i, j = general matrix indexing
    // l = layer
    unsigned e;
    long long l;
    size_t r, i, j;
    double h;
    double loss;

    if (df.n_features - 1 != nn._layers[0].nneurons) {
        fprintf(
            stderr,
            "%s: Input vector size error (df.n_features - 1 = %zu) != (nn.input_layer.nneurons = %zu)\n",
            __func__, df.n_features - 1, nn._layers[0].nneurons
        );
        exit(EXIT_FAILURE);
    }

    for (e = 0; e < epochs; e++) {
        loss = 0.0;

        // Allocate memory to hold average changes
        neuron **n_changes = (neuron **) malloc(nn.nlayers * sizeof(neuron *));
        MALLOC_CHECK(n_changes);
        for (i = 0; i < nn.nlayers; i++) {
            n_changes[i] = (neuron *) calloc(sizeof(neuron), nn._layers[i].nneurons);
            MALLOC_CHECK(n_changes[i]);
        }

        double **w_changes = (double **) malloc((nn.nlayers - 1) * sizeof(double *));
        MALLOC_CHECK(w_changes);
        for (i = 0; i < nn.nlayers - 1; i++) {
            w_changes[i] = (double *) calloc(sizeof(double), nn._layers[i].nneurons * nn._layers[i + 1].nneurons);
            MALLOC_CHECK(w_changes[i]);
        }


        for (r = 0; r < df.n_samples; r++) {
            // Perform for each sample

            /* ======================= Forward Pass ======================= */

            for (i = 0; i < df.n_features - 1; i++) {
                // Set the initial values

                nn._layers[0].neurons[i].value = df.data[r][i];
            }

            for (l = 0; l < nn.nlayers - 1; l++) {
                // Calculate for a layer (this is actually input for layer l + 1)

                for (i = 0; i < nn._layers[l + 1].nneurons; i++) {
                    // Calculate for one neuron (neuron.value)

                    // h = g(sum(wx) + b)
                    h = nn._layers[l + 1].neurons[i].bias;
                    for (j = 0; j < nn._layers[l].nneurons; j++) {
                        // For a single connection \
                           (connection[l][j][i] = connection from neuron[j] of layer l to neuron[i] of l + 1) \
                           j = rows, i = columns

                        h += nn._layers[l].neurons[j].value * nn._connections[l][MATINDEX(j, i, nn._layers[l + 1].nneurons)];
                    }

                    if (l != nn.nlayers - 2)
                        nn._layers[l + 1].neurons[i].value = ACTIVATION(h);
                    else
                        nn._layers[l + 1].neurons[i].value = h;
                }

                // Calculate for the next layer using values of the previous layers (for loop takes care of this)
            }

            // Apply softmax to the final output layer (y_hat)
            softmax(nn._layers[nn.nlayers - 1].nneurons, nn._layers[nn.nlayers - 1].neurons);

            // Calculate the loss for each sample (LOSS = Categorical Cross Entropy)
            // Assume label encoding for the final output (0 to nclasses - 1)
            loss += -log(nn._layers[nn.nlayers - 1].neurons[(size_t) df.data[r][df.n_features - 1]].value);
        

            /* ======================= Backward Pass ======================= */

            // For each sample this value of the neuron which we save does not matter so we can reset it to zero
            for (l = 0; l < nn.nlayers; l++)
                for (i = 0; i < nn._layers[l].nneurons; i++)
                    n_changes[l][i].value = 0.0;

            double *softmax_cce_deriv = (double *) malloc(nn._layers[nn.nlayers - 1].nneurons * sizeof(double));
            MALLOC_CHECK(softmax_cce_deriv);
            for (i = 0; i < nn._layers[nn.nlayers - 1].nneurons; i++)
                softmax_cce_deriv[i] = nn._layers[nn.nlayers - 1].neurons[i].value;
            softmax_cce_deriv[(size_t) df.data[r][df.n_features - 1]] -= 1.0;

            // Update the bias of the neurons of the final layer
            for (i = 0; i < nn._layers[nn.nlayers - 1].nneurons; i++)
                n_changes[nn.nlayers - 1][i].bias += softmax_cce_deriv[i];

            // Update the weights and values (backpropagation) of the neurons from the second last layer
            for (i = 0; i < nn._layers[nn.nlayers - 1].nneurons; i++) {
                for (j = 0; j < nn._layers[nn.nlayers - 2].nneurons; j++) {
                    w_changes[nn.nlayers - 2][MATINDEX(j, i, nn._layers[nn.nlayers - 1].nneurons)] \
                        += softmax_cce_deriv[i] * nn._layers[nn.nlayers - 2].neurons[j].value;
                    n_changes[nn.nlayers - 2][j].value \
                        += softmax_cce_deriv[i] * nn._connections[nn.nlayers - 2][MATINDEX(j, i, nn._layers[nn.nlayers - 1].nneurons)];
                }
            }

            // Update the weights, biases and backpropagate for the rest of the layers
            for (l = nn.nlayers - 3; l >= 0; l--) {
                for (i = 0; i < nn._layers[l + 1].nneurons; i++) {
                    // For every neuron in the next layer
                    
                    // Update the bias of the neurons of next layer
                    double change_bias = -(n_changes[l + 1][i].value / nn._layers[l + 1].neurons[i].value) * ACTIVATION_DERIV(nn._layers[l + 1].neurons[i].value);
                    n_changes[l + 1][i].bias += change_bias;

                    // Update the weights of the connections from this layer to that neuron and backpropagate
                    for (j = 0; j < nn._layers[l].nneurons; j++) {
                        w_changes[l][MATINDEX(j, i, nn._layers[l + 1].nneurons)] \
                            += change_bias * nn._layers[l].neurons[j].value;
                        n_changes[l][j].value \
                            += change_bias * nn._connections[l][MATINDEX(j, i, nn._layers[l + 1].nneurons)];
                    }
                }

                // This will be handled for each layer by the loop
            }


            free(softmax_cce_deriv);
        }

        // Average changes over each sample
        // For a single sample, sum over the changes required by a single neuron \
           to the previous layer's weights and biases
        for (l = 0; l < nn.nlayers; l++) {
            for (i = 0; i < nn._layers[l].nneurons; i++) {
                nn._layers[l].neurons[i].bias -= a * (n_changes[l][i].bias / (2.0 * df.n_samples));
            }
        }

        for (l = 0; l < nn.nlayers - 1; l++) {
            for (j = 0; j < nn._layers[l].nneurons; j++) {
                for (i = 0; i < nn._layers[l + 1].nneurons; i++) {
                    nn._connections[l][MATINDEX(j, i, nn._layers[l + 1].nneurons)] -= a * (w_changes[l][MATINDEX(j, i, nn._layers[l + 1].nneurons)] / (2 * df.n_samples));
                }
            }
        }

        // Free the allocated memory for the changes
        for (i = 0; i < nn.nlayers; i++) {
            free(n_changes[i]);
        }
        free(n_changes);

        for (i = 0; i < nn.nlayers - 1; i++) {
            free(w_changes[i]);
        }
        free(w_changes);

        loss /= df.n_samples;
        printf("[%u/%u] Current loss: %lf, threshold: %lf\r", e, epochs, loss, th);

        if (loss < th) {
            puts("Good enough loss achieved stopping early                                    ");
            goto trained;
        }
    }

    trained:
        return;
}

double *test_model(const model nn, const DataFrame df)
{
    if (df.n_features - 1 != nn._layers[0].nneurons) {
        fprintf(
            stderr,
            "%s: Input vector size error (df.n_features - 1 = %zu) != (nn.input_layer.nneurons = %zu)\n",
            __func__, df.n_features - 1, nn._layers[0].nneurons
        );
        exit(EXIT_FAILURE);        
    }
    
    double *y = (double *) calloc(sizeof(double), df.n_samples);
    MALLOC_CHECK(y);

    long long l;
    size_t r, i, j;
    double h;
    double max;

    for (r = 0; r < df.n_samples; r++) {
        // For each sample

        h = 0.0;

        for (i = 0; i < df.n_features - 1; i++) {
            // Set the initial values

            nn._layers[0].neurons[i].value = df.data[r][i];
        }

        /* Forward Pass */
        for (l = 0; l < nn.nlayers - 1; l++) {
            // Calculate for a layer (this is actually input for layer l + 1)

            for (i = 0; i < nn._layers[l + 1].nneurons; i++) {
                // Calculate for one neuron (neuron.value)

                // h = g(sum(wx) + b)
                h = nn._layers[l + 1].neurons[i].bias;
                for (j = 0; j < nn._layers[l].nneurons; j++) {
                    // For a single connection \
                        (connection[l][j][i] = connection from neuron[j] of layer l to neuron[i] of l + 1) \
                        j = rows, i = columns

                    h += nn._layers[l].neurons[j].value * nn._connections[l][MATINDEX(j, i, nn._layers[l + 1].nneurons)];
                }

                if (l != nn.nlayers - 2)
                    nn._layers[l + 1].neurons[i].value = ACTIVATION(h);
                else
                    nn._layers[l + 1].neurons[i].value = h;
            }

            // Calculate for the next layer using values of the previous layers (for loop takes care of this)
        }

        // Apply softmax to the final output layer (y_hat)
        softmax(nn._layers[nn.nlayers - 1].nneurons, nn._layers[nn.nlayers - 1].neurons);

        // No need to initialise y since it is already calloced
        for (i = 0; i < nn._layers[nn.nlayers - 1].nneurons; i++)
            if (nn._layers[nn.nlayers - 1].neurons[(size_t) y[r]].value < nn._layers[nn.nlayers - 1].neurons[i].value) {
                y[r] = i;
            }
    }

    return y;
}

double accuracy_metric(DataFrame df, double *y)
{
    CHECK_PARAM(y);

    size_t r;
    unsigned sum = 0;
    for (r = 0; r < df.n_samples; r++)
    {
        if ((unsigned) y[r] == (unsigned) df.data[r][df.n_features - 1])
            sum++;
    }

    return (double) sum / df.n_samples;
}

void free_model(model nn)
{
    size_t i = 0;

    // Free the _nconnections array
    free(nn._nconnections);
    nn._nconnections = NULL;

    for (i = 0; i < nn.nlayers - 1; i++) {
        // Free the connections matrix between two layers
        free(nn._connections[i]);
        nn._connections[i] = NULL;
    }

    for (i = 0; i < nn.nlayers; i++) {
        // Free the neurons of a single layer
        free(nn._layers[i].neurons);
        nn._layers[i].neurons = NULL;
    }

    free(nn._layers);
    nn._layers = NULL;
    return;
}

void free_dataframe(DataFrame df)
{
    free(df.data);
    df.data = NULL;
}

static inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

static inline double sigmoid_deriv(double x)
{
    return (x) * (1 - x);
}

static inline double relu(double x)
{
    return (x > 0) ? x : 0;
}

static inline double relu_deriv(double x)
{
    return (x > 0) ? 1 : 0;
}

static inline void softmax(size_t n, neuron *ns)
{
    CHECK_PARAM(ns);

    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += exp(ns[i].value);
    for (size_t i = 0; i < n; i++)
        ns[i].value = exp(ns[i].value) / sum;
}
