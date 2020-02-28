/*
 * nn_xor.c
 *
 *  Created on: 26 февр. 2020 г.
 *      Author: mikhail
 */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <time.h> //clock

const float RAND_MAX_F = RAND_MAX;

struct neuron{
	float out;
	float *in;
	float *w;
	float *dw;
	float (*fActive)(float);
	uint16_t numIn;
};
struct vector{
	float *x;
	float *z;
};
struct network{
	int numInput;
	int numHidden;
	int numOutput;
	int num_w1;	//numbers of synapse on 1 layer
	float *w1;	//weight of synapse on 1 layer
	float *valueActivation1; //value of activation function of 1 layer
	float *df1;	//diff for neurons of 1 layer
	float *delta1;
	int num_w2;	//numbers of synapse on 2 layer
	float *w2;	//weight of synapse on 2 layer
	float *valueActivation2; //value of activation function of 2 layer
	float *df2;	//diff for neurons of 2 layer
	float *delta2;

	void (*fTrain)(float *X, float *Y, float speedLearn, float thresholdErr, float maxEpoch);
};
float fActivationSigmoid(float inData){
	return (1 / (1 + exp(-inData)));
}
void forwardPropagation(struct network *NN, float *X){
	int i,j;
	for (i = 0; i<(NN->numHidden); i++){
		NN->valueActivation1[i] = 0;
		for (j = 0; j<(NN->numInput); j++){
			NN->valueActivation1[i] += NN->w1[NN->numInput * i + j] * X[j];
		}
		NN->valueActivation1[i] = fActivationSigmoid(NN->valueActivation1[i]);
	}

	for (i = 0; i<(NN->numOutput); i++){
		NN->valueActivation2[i] = 0;
		for (j = 0; j<(NN->numHidden); j++){
			NN->valueActivation2[i] += NN->w2[NN->numHidden * i + j] * NN->valueActivation1[j];
		}
		NN->valueActivation2[i] = fActivationSigmoid(NN->valueActivation2[i]);
	}
	//Calculate of diff
	for (i = 0; i<(NN->numHidden); i++){
		NN->df1[i] = (1 - NN->valueActivation1[i]) * NN->valueActivation1[i];
	}

	for (i = 0; i<(NN->numOutput); i++){
		NN->df2[i] = (1 - NN->valueActivation2[i]) * NN->valueActivation2[i];
	}
}

void backwardPropagation(struct network *NN, float *Y, float *err){
	int i,j;

	for (i = 0; i<(NN->numOutput); i++){
		NN->delta2[i] = NN->df2[i] * (Y[i] - NN->valueActivation2[i]);
		*err = (Y[i] - NN->valueActivation2[i]) * (Y[i] - NN->valueActivation2[i]) / 2;
	}

	for (i = 0; i<(NN->numHidden); i++){
		NN->delta1[i] = 0;
		for (j = 0; j<(NN->numOutput); j++){
			NN->delta1[i] += NN->w2[NN->numHidden * j + i] * NN->delta2[j];
		}
		NN->delta1[i] *= NN->df1[i];
	}
}
void updateWeights(struct network *NN, float speedLearn, float moment, float *X){
	int i,j;

	for (i = 0; i<(NN->numHidden); i++){
		for (j = 0; j<(NN->numOutput); j++){
			NN->w2[i + j * (NN->numHidden)] += speedLearn * NN->delta2[j] * NN->valueActivation1[i] + moment*NN->delta2[j];
		}
	}

	for (i = 0; i<(NN->numInput); i++){
		for (j = 0; j<(NN->numHidden); j++){
			NN->w1[i + j * (NN->numInput)] += speedLearn * NN->delta1[j] * X[i] + moment*NN->delta1[j];
		}
	}
}

void Train(struct  network *NN, float *X, float *Y, int length, float speedLearn, float moment, float thresholdErr, float maxEpoch){
	int epoch, i;
	float errEpoch;
	for (epoch = 0; epoch < maxEpoch; epoch++){
		errEpoch = 0;
		for (i = 0; i < length; i++){
			forwardPropagation(NN, &X[2*i]);
			backwardPropagation(NN, &Y[i], &errEpoch);
			updateWeights(NN, speedLearn, moment, &X[2*i]);
		}
		printf("Epoch: %d, error: %1.7f\n", epoch, errEpoch);
		if (errEpoch < thresholdErr)
			return;
	}
}




float fActivationTanh(float inData){
	return ((exp(2 * inData) - 1)/(exp(2 * inData) + 1));
}


void initNeuron(struct neuron *pNeur, uint16_t numIn, float (*fptr)(float));

void createNetwork(struct network *NN, int numInput, int numHidden, int numOutput){
	int i;

	NN->numInput = numInput;
	NN->numHidden = numHidden;
	NN->numOutput = numOutput;

	NN->num_w1 	= numInput * numHidden;
	NN->w1 	= malloc(NN->num_w1 * sizeof(float));
	NN->valueActivation1 = malloc(numHidden * sizeof(float));
	NN->df1 = malloc(numHidden * sizeof(float));
	NN->delta1 = malloc(numHidden * sizeof(float));

	NN->num_w2 	= numOutput * numHidden;
	NN->w2	= malloc(NN->num_w2 * sizeof(float));
	NN->valueActivation2 = malloc(numOutput * sizeof(float));
	NN->df2 = malloc(numOutput * sizeof(float));
	NN->delta2 = malloc(numHidden * sizeof(float));

	for (i = 0; i < NN->num_w1; i++){
		NN->w1[i] = rand() / RAND_MAX_F;
	}
	for (i = 0; i < NN->num_w2; i++){
		NN->w2[i] = rand() / RAND_MAX_F;
	}
}

int main(int argc, char *argv[]){
	float X[8] = {0, 0, 0, 1, 1, 0, 1, 1};
	float Y[4] = {0, 0, 0, 1};
	struct network NN;

	createNetwork(&NN,2, 3, 1);
	Train(&NN, X, Y, 4, 0.1, 0.01, 0.0000001, 100000);
	float in1, in2, out, err;
	float E = 0.01;
	float A = 0.3;
	struct neuron *pNeur0 = malloc(sizeof(struct neuron));
	struct neuron *pNeur1 = malloc(sizeof(struct neuron));
	struct neuron *pNeur2 = malloc(sizeof(struct neuron));
	struct neuron *pNeur3 = malloc(sizeof(struct neuron));
	initNeuron(pNeur0, 2, fActivationSigmoid);
	initNeuron(pNeur1, 2, fActivationSigmoid);
	initNeuron(pNeur2, 3, fActivationSigmoid);
	initNeuron(pNeur3, 2, fActivationSigmoid);

	float setX1[4] 	= {0, 1, 0, 1};
	float setX2[4] 	= {0, 0, 1, 1};
	float setY[4]	= {0, 1, 1, 0};

	in1	= 1;
	in2 = 0;
	out = 1;

	pNeur0->w[0] = 0.45;
	pNeur0->w[1] = -0.12;
	pNeur1->w[0] = 0.78;
	pNeur1->w[1] = 0.13;
	pNeur2->w[0] = 1.5;
	pNeur2->w[1] = -2.3;
	pNeur2->w[2] = 0.8;
	pNeur3->w[0] = 0.2;
	pNeur3->w[1] = -1;

	int Epoha, i;
	int trainSet = 4;

	for (Epoha = 0; Epoha < 100; Epoha++){
		err = 0;
		for (i = 0; i < trainSet; i++){
			in1 = setX1[i];
			in2 = setX2[i];
			out = setY[i];
			pNeur0->out = pNeur0->fActive(pNeur0->w[0] * in1 + pNeur0->w[1] * in2);
			pNeur1->out = pNeur1->fActive(pNeur1->w[0] * in1 + pNeur1->w[1] * in2);
			pNeur3->out = pNeur1->fActive(pNeur3->w[0] * in1 + pNeur3->w[1] * in2);
			pNeur2->out = pNeur2->fActive(pNeur2->w[0] * pNeur0->out + pNeur2->w[1] * pNeur1->out);// + pNeur2->w[2] * pNeur3->out);

			err += (1 - pNeur2->out)*(1 - pNeur2->out) / 1;
			//printf("Err %1.3f\n",err);




			//printf("w1 w2 w3 w4 w5 w6\n");
			//printf("%1.3f %1.3f %1.3f %1.3f %1.3f %1.3f\n",pNeur0->w[0], pNeur1->w[0], pNeur0->w[1], pNeur1->w[1], pNeur2->w[0], pNeur2->w[1]);
		}
		float deltaO, deltaH1, deltaH2, deltaH3;
		float GRADw1, GRADw2, GRADw3, GRADw4, GRADw5, GRADw6, GRADw7, GRADw8, GRADw9;

		deltaO = (out - pNeur2->out) * ((1 - pNeur2->out) * pNeur2->out);
		deltaH1 = (1 - pNeur0->out) * pNeur0->out * (pNeur2->w[0] * deltaO);
		GRADw5 = pNeur1->out * deltaO;
		pNeur2->dw[0] = (E * GRADw5 + A * pNeur2->dw[0]);
		pNeur2->w[0] = pNeur2->w[0] + pNeur2->dw[0];

		deltaH2 = (1 - pNeur1->out) * pNeur1->out * (pNeur2->w[1] * deltaO);
		GRADw6 = pNeur0->out * deltaO;
		pNeur2->dw[1] = (E * GRADw6 + A * pNeur2->dw[1]);
		pNeur2->w[1] = pNeur2->w[1] + pNeur2->dw[1];

		deltaH3 = (1 - pNeur3->out) * pNeur3->out * (pNeur2->w[2] * deltaO);
		GRADw7 = pNeur3->out * deltaO;
		pNeur2->dw[2] = (E * GRADw7 + A * pNeur2->dw[2]);
		pNeur2->w[2] = pNeur2->w[2] + pNeur2->dw[2];

		GRADw1 = in1 * deltaH1;
		GRADw2 = in1 * deltaH2;
		GRADw3 = in2 * deltaH1;
		GRADw4 = in2 * deltaH2;
		GRADw8 = in1 * deltaH3;
		GRADw9 = in2 * deltaH3;

		pNeur0->dw[0] = (E * GRADw1 + A * pNeur0->dw[0]);
		pNeur0->w[0] = pNeur0->w[0] + pNeur0->dw[0];
		pNeur0->dw[1] = (E * GRADw3 + A * pNeur0->dw[1]);
		pNeur0->w[1] = pNeur0->w[1] + pNeur0->dw[1];
		pNeur1->dw[0] = (E * GRADw2 + A * pNeur1->dw[0]);
		pNeur1->w[0] = pNeur1->w[0] + pNeur1->dw[0];
		pNeur1->dw[1] = (E * GRADw4 + A * pNeur1->dw[1]);
		pNeur1->w[1] = pNeur1->w[1] + pNeur1->dw[1];

		pNeur3->dw[0] = (E * GRADw8 + A * pNeur3->dw[0]);
		pNeur3->w[0] = pNeur3->w[0] + pNeur3->dw[0];
		pNeur3->dw[1] = (E * GRADw9 + A * pNeur3->dw[1]);
		pNeur3->w[1] = pNeur3->w[1] + pNeur3->dw[1];
	}
	return 0;
}

void initNeuron(struct neuron *pNeur, uint16_t numIn, float (*fptr)(float)){
	pNeur->numIn = numIn;
	pNeur->in 	= malloc(pNeur->numIn * sizeof(float));
	pNeur->w 	= malloc(pNeur->numIn * sizeof(float));
	pNeur->fActive = fptr;
	pNeur->dw	= malloc(pNeur->numIn * sizeof(float));
}
