#pragma once
#include <Eigen/Dense>
#include "Functions.h"
#include <vector>

class Layer
{
private:
	Eigen::MatrixXf* actiVal;
	Eigen::MatrixXf* weights;
	Eigen::MatrixXf* transposedWeights;
	Eigen::MatrixXf* gradientBuffer; //pls don't allocate memory and de-alocate it at runtime, do the same for other required matrices 
	Eigen::MatrixXf* deltaBuffer;
	Eigen::MatrixXf* deltaBufferB;
    //Eigen::MatrixXf* deltaHelperBuffer;
	Eigen::MatrixXf* zBuffer;
	Eigen::MatrixXf* zBufferB;
	std::function<float(float)> activFunc;
	std::function<float(float)> activFuncGrad;
	bool outputTag;
	int n_neurons;
	int next_n_neurons;

public:
	Layer();
	Layer(int n_neurons, int next_n_neurons, bool outputTag = false, float(*activFunc)(float) = Functions::Linear::f, float(*activFuncGrad)(float) = Functions::Linear::df);


	void to_next_layer(Layer* target);
	void update();
	void compute_z(Layer* target); //Might be useless
	void compute_delta(Layer* target, int classIdx = -1);
	void compute_delta_new(Layer* target, Eigen::MatrixXf* output = nullptr);

	//Setters
	void set_actiVal(Eigen::MatrixXf* new_actiVal);
	void set_weights(Eigen::MatrixXf* new_weights);
	void set_gradient(Eigen::MatrixXf* new_gradient);

	//Getters
	int get_n_neurons();
	bool get_output_flag();
	std::function<float(float)> get_activFunc();
	Eigen::MatrixXf* get_actiVal();
	Eigen::MatrixXf* get_weights();
	Eigen::MatrixXf* get_transposed_weights();
	Eigen::MatrixXf* get_z(); 
	Eigen::MatrixXf* get_z_b();
	Eigen::MatrixXf* get_delta();
	Eigen::MatrixXf* get_delta_b();
	Eigen::MatrixXf* get_gradient();
	//Eigen::MatrixXf* get_helper_delta_buffer();
	

	//tests
	float test_activFunc(float x);
	void print_weights();
	void print_actiVal();
};

class NeuralNetwork
{
private:
	std::vector<Layer*>* layers;
	int nLayers;
	float learningRate;
	float regParam;
	int nIter;
	int batchSize;

	void forward_propagation(Eigen::MatrixXf* inputs);
	void compute_delta(Layer* layer, int classIdx = 0, Eigen::MatrixXf* z = nullptr, Eigen::MatrixXf* nextDelta = nullptr);


public:
	NeuralNetwork(std::vector<Layer*>* _layers, float _learningRate = 1.f, float _regParam = 0.f, int _nIter = 1, int _batchSize = 1);

	void assimilate_classif(Eigen::MatrixXf* input, int result);
	void assimilate_regress(Eigen::MatrixXf* input, Eigen::MatrixXf* output, int inputSize);
	void fit_classif(Eigen::MatrixXf* inputs, Eigen::MatrixXf* results);
	void fit_regress(Eigen::MatrixXf* inputs, Eigen::MatrixXf* results);

	int predict_classif(Eigen::MatrixXf* input);
	Eigen::MatrixXf* predict_regress(Eigen::MatrixXf* input);

	void set_learning_rate(float _learningRate);
	void set_reg_param(float _regParam);
	float get_learning_rate();
	float get_reg_param();
	Eigen::MatrixXf* get_weights(size_t idx);

};

