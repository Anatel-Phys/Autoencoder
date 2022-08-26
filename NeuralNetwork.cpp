#include "NeuralNetwork.h"
#include "Functions.h"
#include <iostream>

Layer::Layer()
{
	this->n_neurons = 1;
	this->next_n_neurons = 1;

	this->outputTag = false;
	this->weights = new Eigen::MatrixXf(Eigen::MatrixXf::Random(1,1));
	this->actiVal = new Eigen::MatrixXf(1, 1);
	this->deltaBuffer = new Eigen::MatrixXf;

	this->activFunc = Functions::Linear::f;
	this->activFuncGrad = Functions::Linear::df;
}

Layer::Layer(int _n_neurons, int _next_n_neurons, bool _outputTag, float(*_activFunc)(float), float(*_activFuncGrad)(float)) 
{
	this->n_neurons = _n_neurons;
	this->next_n_neurons = _next_n_neurons;
	
	this->outputTag = _outputTag;
	this->actiVal = new Eigen::MatrixXf(Eigen::MatrixXf::Zero(_n_neurons + 1 * (!_outputTag), 1));
	if (!_outputTag)
		this->actiVal->operator()(0, 0) = 1.f;
	
	this->deltaBufferB= new Eigen::MatrixXf(Eigen::MatrixXf::Zero(_n_neurons + !_outputTag, 1));
	this->deltaBuffer = new Eigen::MatrixXf(Eigen::MatrixXf::Zero(_n_neurons, 1));
	this->zBufferB = new Eigen::MatrixXf(Eigen::MatrixXf::Zero(_n_neurons + 1, 1));
	this->zBuffer = new Eigen::MatrixXf(Eigen::MatrixXf::Zero(_n_neurons, 1));

	this->weights = new Eigen::MatrixXf(Eigen::MatrixXf::Random(_next_n_neurons, _n_neurons + 1));
	this->transposedWeights = new Eigen::MatrixXf(this->weights->transpose());
	this->gradientBuffer = new Eigen::MatrixXf(Eigen::MatrixXf::Zero(_next_n_neurons, _n_neurons + 1));

	this->activFunc = _activFunc;
	this->activFuncGrad = _activFuncGrad;
}


void Layer::to_next_layer(Layer* target)
{
	if (this->next_n_neurons != target->get_n_neurons())
		throw "Layers size not matching";
	Eigen::MatrixXf product = *this->weights * *this->actiVal;
	*target->get_z() = product;
	target->get_z_b()->block(1, 0, target->get_n_neurons(), 1) = product;
	target->get_z_b()->operator()(0, 0) = 1.f;
	product = product.unaryExpr(this->activFunc);
	target->set_actiVal(&product);
}

void Layer::update()
{
	*this->weights -= *this->gradientBuffer;
	*this->transposedWeights = this->weights->transpose();
}

void Layer::compute_z(Layer* target) //set the z of the next layer (target)
{
	*target->get_z() = *this->weights * *this->actiVal;
	target->get_z_b()->operator()(0, 0) = 1.f;
	target->get_z_b()->block(1, 0, target->get_n_neurons(), 1) = *target->get_z();
}

void Layer::compute_delta(Layer* target, int classIdx) //compute the delta of this layer with the next layer's delta
{
	if (this->outputTag)
		*this->deltaBuffer = *this->actiVal - Functions::Helper::logic_vec(classIdx, this->n_neurons);
	else
	{
		*this->deltaBufferB = *this->transposedWeights * *target->get_delta();
		*this->deltaBufferB = this->deltaBufferB->cwiseProduct(this->actiVal->unaryExpr(this->activFuncGrad));
		*this->deltaBuffer = this->deltaBufferB->block(1, 0, this->n_neurons, 1);
	}
}

void Layer::compute_delta_new(Layer* target, Eigen::MatrixXf* output) //compute the delta of this layer with the next layer's delta
{
	if (this->outputTag)
		*this->deltaBuffer = *this->actiVal - *output;
	else
	{
		*this->deltaBufferB = *this->transposedWeights * *target->get_delta();
		*this->deltaBufferB = this->deltaBufferB->cwiseProduct(this->actiVal->unaryExpr(this->activFuncGrad));
		*this->deltaBuffer = this->deltaBufferB->block(1, 0, this->n_neurons, 1);
	}
}

void Layer::set_actiVal(Eigen::MatrixXf* new_actiVal)
{
	if (this->outputTag) {
		if (this->actiVal->rows() != new_actiVal->rows() || new_actiVal->cols() != 1)
			throw "Error when assigning new activation values. Expected a column vector of size " + std::to_string(this->n_neurons) + " x 1, but got an array of size " + std::to_string(new_actiVal->rows());
		*this->actiVal = *new_actiVal;
	}
	else {
		if (this->actiVal->rows() != new_actiVal->rows() + 1 || new_actiVal->cols() != 1)
			throw "Error when assigning new activation values. Expected a column vector of size " + std::to_string(this->n_neurons) + " x 1, but got an array of size " + std::to_string(new_actiVal->rows());
		(*this->actiVal).block(1,0,n_neurons, 1) = *new_actiVal;
	}
}

void Layer::set_weights(Eigen::MatrixXf* new_weights)
{
	if (new_weights->cols() != this->weights->cols() || new_weights->rows() != this->weights->rows())
		throw;
	*this->weights = *new_weights;
}

void Layer::set_gradient(Eigen::MatrixXf* new_gradient)
{
	if (new_gradient->cols() != this->weights->cols() || new_gradient->rows() != this->weights->rows())
		throw;
	*this->gradientBuffer = *new_gradient;
}

int Layer::get_n_neurons()
{
	return this->n_neurons;
}

bool Layer::get_output_flag()
{
	return this->outputTag;
}

std::function<float(float)> Layer::get_activFunc()
{
	return this->activFunc;
}

Eigen::MatrixXf* Layer::get_actiVal()
{
	return this->actiVal;
}

Eigen::MatrixXf* Layer::get_weights()
{
	return this->weights;
}

Eigen::MatrixXf* Layer::get_transposed_weights()
{
	return this->transposedWeights;
}

Eigen::MatrixXf* Layer::get_z()
{
	return this->zBuffer;
}

Eigen::MatrixXf* Layer::get_z_b()
{
	return this->zBufferB;
}

Eigen::MatrixXf* Layer::get_delta()
{
	return this->deltaBuffer;
}

Eigen::MatrixXf* Layer::get_delta_b()
{
	return this->deltaBufferB;
}

Eigen::MatrixXf* Layer::get_gradient()
{
	return this->gradientBuffer;
}

//Eigen::MatrixXf* Layer::get_helper_delta_buffer()
//{
//	return this->deltaHelperBuffer;
//}

float Layer::test_activFunc(float x)
{
	return this->activFunc(x);
}

void Layer::print_weights()
{
	std::cout << *this->weights << std::endl;
}

void Layer::print_actiVal()
{
	std::cout << *this->actiVal << std::endl;
}

void NeuralNetwork::forward_propagation(Eigen::MatrixXf* inputs)
{
	this->layers->at(0)->set_actiVal(inputs);

	for (size_t i = 0; i < this->layers->size() - 1; i++)
		this->layers->at(i)->to_next_layer(this->layers->at(i + 1));
}

NeuralNetwork::NeuralNetwork(std::vector<Layer*>* _layers, float _learningRate, float _regParam, int _nIter, int _batchSize)
{
	this->layers = new std::vector<Layer*>(*_layers);
	this->nLayers = this->layers->size();
	this->learningRate = _learningRate;
	this->regParam = _regParam;
	this->nIter = _nIter;
	this->batchSize = _batchSize;	
}


void NeuralNetwork::assimilate_classif(Eigen::MatrixXf* input, int result)
{
	this->forward_propagation(input); //Computes the z along with the final output
	Eigen::MatrixXf* classif_output = new Eigen::MatrixXf(this->layers->at(this->nLayers - 1)->get_n_neurons(), 1);
	*classif_output = Functions::Helper::logic_vec(result, this->layers->at(static_cast<int>(this->nLayers) - 1)->get_n_neurons());
	this->layers->at(static_cast<int>(this->nLayers) - 1)->compute_delta_new(nullptr, classif_output);
	for (size_t i = this->nLayers - 2; i > 0; i--)
	{
		this->layers->at(i)->compute_delta(this->layers->at(i + 1));
	}

	for (size_t i = 0; i < static_cast<int>(this->nLayers) - 1; i++)
	{
		*this->layers->at(i)->get_gradient() = *this->layers->at(i + 1)->get_delta() * Eigen::MatrixXf(this->layers->at(i)->get_actiVal()->transpose());
		*this->layers->at(i)->get_gradient() = this->learningRate * this->layers->at(i)->get_gradient()->array();
	}
	//no regularization for now

	for (size_t i = 0; i < static_cast<int>(this->nLayers) - 1; i++)
		this->layers->at(i)->update();
}

void NeuralNetwork::assimilate_regress(Eigen::MatrixXf* input, Eigen::MatrixXf* output, int inputSize)
{
	this->forward_propagation(input); //Computes the z along with the final output
	this->layers->at(static_cast<int>(this->nLayers) - 1)->compute_delta_new(nullptr, output);
	for (size_t i = this->nLayers - 2; i > 0; i--)
	{
		this->layers->at(i)->compute_delta(this->layers->at(i + 1));
	}

	for (size_t i = 0; i < static_cast<int>(this->nLayers) - (int)1; i++)
	{
		*this->layers->at(i)->get_gradient() = *this->layers->at(i + 1)->get_delta() * Eigen::MatrixXf(this->layers->at(i)->get_actiVal()->transpose());
		*this->layers->at(i)->get_gradient() = (this->learningRate / inputSize) * this->layers->at(i)->get_gradient()->array();
	}

	//for (size_t i = 0; i < static_cast<int>(this->nLayers) - 1; i++)
	//	this->layers->at(i)->update();
}

void NeuralNetwork::fit_classif(Eigen::MatrixXf* inputs, Eigen::MatrixXf* results)
{
	size_t inputSize = inputs->rows();
	Eigen::MatrixXf* iBuffer = new Eigen::MatrixXf(inputs->cols(), 1);
	Eigen::MatrixXf* regBuffer;

	for (size_t it = 0; it < this->nIter; it++)
	{
		for (size_t n = 0; n < inputSize; n++)
		{
			*iBuffer = inputs->block(n, 0, 1, inputs->cols()).transpose();
			assimilate_classif(iBuffer, results->operator()(n, 0));

			for (size_t i = 0; i < this->nLayers - 2; i++)
			{
				regBuffer = new Eigen::MatrixXf(*this->layers->at(i)->get_weights());
				//for this particular case I will regularoze bias unit
				//regBuffer->block(0, 0, 1, regBuffer->cols()) = Eigen::MatrixXf::Zero(1, regBuffer->cols()); 
				*regBuffer *= (this->regParam / inputSize);
				*this->layers->at(i)->get_weights() -= *regBuffer;
				delete regBuffer;
			}
		}
		std::cout << "Iteration " << it + 1 << "/" << this->nIter << " done.\n";
	}
}

void NeuralNetwork::fit_regress(Eigen::MatrixXf* inputs, Eigen::MatrixXf* results)
{
	size_t inputSize = inputs->rows();
	Eigen::MatrixXf* iBuffer = new Eigen::MatrixXf(inputs->cols(), 1);
	Eigen::MatrixXf* oBuffer = new Eigen::MatrixXf(results->cols(), 1);
	Eigen::MatrixXf* regBuffer;

	for (size_t it = 0; it < this->nIter; it++)
	{
		for (size_t n = 0; n < inputSize; n++)
		{
			*iBuffer = inputs->block(n, 0, 1, inputs->cols()).transpose();
			*oBuffer = results->block(n, 0, 1, results->cols()).transpose();
			assimilate_regress(iBuffer, oBuffer, inputSize);

			//Maybe the problem is that regularization comes after update and not simultaneously
			for (size_t i = 0; i < this->nLayers - 2; i++)
			{
				regBuffer = new Eigen::MatrixXf(*this->layers->at(i)->get_weights());
				//for this particular case I will regularoze bias unit
				//regBuffer->block(0, 0, 1, regBuffer->cols()) = Eigen::MatrixXf::Zero(1, regBuffer->cols()); 
				*regBuffer *= (this->regParam / inputSize);
				*this->layers->at(i)->get_weights() -= *regBuffer;
				delete regBuffer;
			}

			for (size_t i = 0; i < static_cast<int>(this->nLayers) - 1; i++)
				this->layers->at(i)->update();
		}
		std::cout << "Iteration " << it + 1 << "/" << this->nIter << " done.\n";
	}
}

int NeuralNetwork::predict_classif(Eigen::MatrixXf* input)
{
	forward_propagation(input);
	
	float maxActiVal = this->layers->at(static_cast<int>(this->nLayers) - 1)->get_actiVal()->operator()(0, 0);
	int maxIdx = 0;
	for (size_t i = 1; i < this->layers->at(static_cast<int>(this->nLayers) - 1)->get_n_neurons(); i++)
	{
		if (this->layers->at(static_cast<int>(this->nLayers) - 1)->get_actiVal()->operator()(i, 0) > maxActiVal)
		{
			maxActiVal = this->layers->at(static_cast<int>(this->nLayers) - 1)->get_actiVal()->operator()(i, 0);
			maxIdx = i;
		}
	}

	return maxIdx + 1;
}

Eigen::MatrixXf* NeuralNetwork::predict_regress(Eigen::MatrixXf* input) //really need to decide if input data should be row or column
{
	forward_propagation(input);

	Eigen::MatrixXf* output = new Eigen::MatrixXf(this->layers->at(static_cast<int>(this->nLayers) - 1)->get_n_neurons(), 1);
	*output = this->layers->at(static_cast<int>(this->nLayers) - 1)->get_actiVal()->transpose();
	
	return output;
}

void NeuralNetwork::set_learning_rate(float _learningRate)
{
	this->learningRate = _learningRate;
}

void NeuralNetwork::set_reg_param(float _regParam)
{
	this->regParam = _regParam;
}

float NeuralNetwork::get_learning_rate()
{
	return this->learningRate;
}

float NeuralNetwork::get_reg_param()
{
	return this->regParam
		;
}

Eigen::MatrixXf* NeuralNetwork::get_weights(size_t idx)
{
	if (idx >= this->nLayers)
		return nullptr;
	return this->layers->at(idx)->get_weights();
}

