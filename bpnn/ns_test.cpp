#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "backprop.hpp"

void train(ano::bpnn::NN *nn)
{
    int n = 1000;
    double **trainingSet = new double *[n];
    for (int i = 0; i < n; i++)
    {
        trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];

        bool classA = i % 2;

        for (int j = 0; j < nn->n[0]; j++)
        {
            if (classA)
            {
                trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX) + 0.6;
            }
            else
            {
                trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX) + 0.2;
            }
        }

        trainingSet[i][nn->n[0]] = (classA) ? 1.0 : 0.0;
        trainingSet[i][nn->n[0] + 1] = (classA) ? 0.0 : 1.0;
    }

    double error = 1.0;
    int i = 0;
    while (error > 0.001)
    {
        ano::bpnn::setInput(nn, trainingSet[i % n]);
        ano::bpnn::feedforward(nn);
        error = ano::bpnn::backpropagation(nn, &trainingSet[i % n][nn->n[0]]);
        i++;
        printf("\rerr=%0.3f", error);
    }
    printf(" (%d iterations)\n", i);

    for (int i = 0; i < n; i++)
    {
        delete[] trainingSet[i];
    }
    delete[] trainingSet;
}

void test(ano::bpnn::NN *nn, int num_samples = 10)
{
    double *in = new double[nn->n[0]];

    int num_err = 0;
    for (int n = 0; n < num_samples; n++)
    {
        bool classA = rand() % 2;

        for (int j = 0; j < nn->n[0]; j++)
        {
            if (classA)
            {
                in[j] = 0.1 * (double)rand() / (RAND_MAX) + 0.6;
            }
            else
            {
                in[j] = 0.1 * (double)rand() / (RAND_MAX) + 0.2;
            }
        }
        printf("predicted: %d\n", !classA);
        ano::bpnn::setInput(nn, in, true);

        ano::bpnn::feedforward(nn);
        int output = ano::bpnn::getOutput(nn, true);
        if (output == classA)
            num_err++;
        printf("\n");
    }
    double err = (double)num_err / num_samples;
    printf("test error: %.2f\n", err);

    delete[] in;
}

int main(int argc, char **argv)
{
    ano::bpnn::NN *nn = ano::bpnn::createNN(2, 4, 2);
    train(nn);

    getchar();

    test(nn, 100);

    getchar();

    ano::bpnn::releaseNN(nn);

    return 0;
}
