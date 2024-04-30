#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <vector>

#include "backprop.hpp"

#define LAMBDA 1.0
#define ETA 0.1

#define SQR(x) ((x) * (x))

void randomize(double *p, int n)
{
	for (int i = 0; i < n; i++)
	{
		p[i] = (double)rand() / (RAND_MAX);
	}
}

NN *createNN(int n, int h, int o)
{
	srand(time(NULL));
	NN *nn = new NN;

	nn->n = new int[3];
	nn->n[0] = n;
	nn->n[1] = h;
	nn->n[2] = o;
	nn->l = 3;

	nn->w = new double **[nn->l - 1];

	for (int k = 0; k < nn->l - 1; k++)
	{
		nn->w[k] = new double *[nn->n[k + 1]];
		for (int j = 0; j < nn->n[k + 1]; j++)
		{
			nn->w[k][j] = new double[nn->n[k]];
			randomize(nn->w[k][j], nn->n[k]);
			// BIAS
			// nn->w[k][j] = new double[nn->n[k] + 1];
			// randomize( nn->w[k][j], nn->n[k] + 1 );
		}
	}

	nn->y = new double *[nn->l];
	for (int k = 0; k < nn->l; k++)
	{
		nn->y[k] = new double[nn->n[k]];
		memset(nn->y[k], 0, sizeof(double) * nn->n[k]);
	}

	nn->in = nn->y[0];
	nn->out = nn->y[nn->l - 1];

	nn->d = new double *[nn->l];
	for (int k = 0; k < nn->l; k++)
	{
		nn->d[k] = new double[nn->n[k]];
		memset(nn->d[k], 0, sizeof(double) * nn->n[k]);
	}

	return nn;
}

void releaseNN(NN *&nn)
{
	for (int k = 0; k < nn->l - 1; k++)
	{
		for (int j = 0; j < nn->n[k + 1]; j++)
		{
			delete[] nn->w[k][j];
		}
		delete[] nn->w[k];
	}
	delete[] nn->w;

	for (int k = 0; k < nn->l; k++)
	{
		delete[] nn->y[k];
	}
	delete[] nn->y;

	for (int k = 0; k < nn->l; k++)
	{
		delete[] nn->d[k];
	}
	delete[] nn->d;

	delete[] nn->n;

	delete nn;
	nn = NULL;
}

void feedforward(NN *nn)
{
	// k - layer index
	// w[layer][to-layer+1][from-layer] - weights
	// y - outputs
	// d - errors
	// n - num of neurons in layers

	auto in = nn->y[0];

	// Propagate through all layers. Start from second layer - inputs are given
	for (int layer = 1; layer < nn->l; layer++)
	{
		auto layer_w = nn->w[layer - 1];
		auto layer_y = nn->y[layer];
		auto layer_y_prev = nn->y[layer - 1];
		auto layer_n = nn->n[layer];
		auto layer_n_prev = nn->n[layer - 1];

		// For every neuron in current layer
		for (int i = 0; i < layer_n; i++)
		{
			// Clear input
			layer_y[i] = 0;

			// Sum up all inputs from previous layer
			for (int j = 0; j < layer_n_prev; j++)
			{
				layer_y[i] += layer_w[i][j] * layer_y_prev[j];
			}
			layer_y[i] = 1.0 / (1.0 + exp(-layer_y[i]));
		}
	}
}

double backpropagation(NN *nn, double *t)
{
	double error = 0.0;

	// Calculate error
	auto n_out = nn->n[nn->l - 1];
	for (int i = 0; i < n_out; i++)
	{
		error += std::pow(nn->out[i] - t[i], 2);
	}
	error /= 2;

	// Calculate deltas for output layer
	for (int i = 0; i < n_out; i++)
	{
		auto layer_d = nn->d[nn->l - 1];
		layer_d[i] = (t[i] - nn->out[i]) * nn->out[i] * (1 - nn->out[i]);
	}

	// Calculate other deltas
	// For every layer except output layer
	for (int layer = nn->l - 2; layer > 0; layer--)
	{
		auto layer_w_next = nn->w[layer]; // This time is layer not layer-1 as we are going backwards
		auto layer_d = nn->d[layer];
		auto layer_d_next = nn->d[layer + 1];
		auto layer_y = nn->y[layer];
		auto layer_n = nn->n[layer];
		auto layer_n_next = nn->n[layer + 1];

		// For every neuron in current layer
		for (int i = 0; i < layer_n; i++)
		{
			// Clear delta
			layer_d[i] = 0;

			// Sum up deltas from next layer
			for (int j = 0; j < layer_n_next; j++)
			{
				layer_d[i] += layer_d_next[j] * layer_w_next[j][i]; // w[layer][j][konst] - iterate through all neurons in next layer
			}

			layer_d[i] *= layer_y[i] * (1 - layer_y[i]);
		}
	}

	// Update weights
	for (int layer = 0; layer < nn->l; layer++)
	{
		auto layer_w = nn->w[layer];
		auto layer_d_next = nn->d[layer + 1];
		auto layer_y = nn->y[layer];
		auto layer_n = nn->n[layer];
		auto layer_n_next = nn->n[layer + 1];

		// Go through all weights in current layer
		for (int i = 0; i < layer_n; i++)
		{
			// Go through all neurons in next layer and calculate Δd and add it to weight
			for (int j = 0; j < layer_n_next; j++)
			{
				auto weight_delta = layer_d_next[j] * layer_y[i];
				layer_w[j][i] += weight_delta;
			}
		}
	}

	return error;
}

void setInput(NN *nn, double *in, bool verbose)
{
	memcpy(nn->in, in, sizeof(double) * nn->n[0]);

	if (verbose)
	{
		printf("input=(");
		for (int i = 0; i < nn->n[0]; i++)
		{
			printf("%0.3f", nn->in[i]);
			if (i < nn->n[0] - 1)
			{
				printf(", ");
			}
		}
		printf(")\n");
	}
}

int getOutput(NN *nn, bool verbose)
{
	double max = 0.0;
	int max_i = 0;
	if (verbose)
		printf(" output=");
	for (int i = 0; i < nn->n[nn->l - 1]; i++)
	{
		if (verbose)
			printf("%0.3f ", nn->out[i]);
		if (nn->out[i] > max)
		{
			max = nn->out[i];
			max_i = i;
		}
	}
	if (verbose)
		printf(" -> %d\n", max_i);
	if (nn->out[0] > nn->out[1] && nn->out[0] - nn->out[1] < 0.1)
		return 2;
	return max_i;
}
