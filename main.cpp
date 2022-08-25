#include <iostream>
#include "NeuralNetwork.h"
#include <Eigen/Dense>
#include <string>
#include <strstream>
#include <fstream>

//DISABLE WARNINGS FOR CIMG LIBRARY
#include <codeanalysis\warnings.h>
#pragma warning( push )
#pragma warning ( disable : ALL_CODE_ANALYSIS_WARNINGS )
#include "CImg.h"
#pragma warning( pop )

Eigen::MatrixXf* load(std::string fileName, int reductionFactor)
{
	std::ifstream file(fileName);

	if (file.is_open())
	{
		std::string bufferLine;
		std::string buffer;
		int dataSize = 0;
		int inputSize = 0;
		Eigen::MatrixXf* target;

		std::getline(file, buffer);
		dataSize++;

		std::stringstream s;
		s << buffer;
		float val;
		while (s >> val)
			inputSize++;

		while (std::getline(file, buffer))
			dataSize++;
		file.seekg(0);
		dataSize = dataSize / reductionFactor;
		target = new Eigen::MatrixXf(dataSize, inputSize);

		std::cout << "Generating matrix of dim " << dataSize << "x" << inputSize << ". (" << inputSize - 1 << " columns of variables and 1 for labels)" << std::endl;
		file.close();
		std::ifstream file(fileName);
		int i = 0;
		int j = 0;
		
		while (std::getline(file, bufferLine) && i < dataSize)
		{
			j = 0;
			s.clear();
			s << bufferLine;
			while (s >> buffer)
			{
				target->operator()(i, j) = std::stoi(buffer);
				j++;
			}
			i++;
		}

		return target;
	}
	else
		return nullptr;

}

//input should be a row vector
cimg_library::CImg<unsigned char> data_to_img(Eigen::MatrixXf* input, size_t x, size_t y)
{
	cimg_library::CImg<unsigned char> image(x, y, 1, 3, 0);
	for (size_t i = 0; i < y; i++)
	{
		for (size_t j = 0; j < x; j++)
		{
			image(i, j, 0) = input->operator()(j * x + i); //needed to swap indices because the image was rotated for some reason
			image(i, j, 1) = input->operator()(j * x + i);
			image(i, j, 2) = input->operator()(j * x + i);
		}
	}

	return image;
}

struct capping
{
	float operator()(float x) const { return std::fmin(x, 255); }
};

int main2()
{
	Eigen::MatrixXf A(3, 3);
	A << 1, 2, 3,
		4, 5, 6,
		7, 8, 9;

	A = A.unaryExpr(capping());
	std::cout << A;

	return 0;
}

int main()
{
	srand(time(0));

	int nData = 6000;
	Eigen::MatrixXf* A = load("data.txt", 60000 / nData);
	Eigen::MatrixXf* data = new Eigen::MatrixXf(A->rows(), 784);
	*data = A->block(0, 0, A->rows(), 784);
	*data /= 255;
	
	Layer encoder1(784, 392, false, Functions::LeakyRelu::f, Functions::LeakyRelu::df);
	Layer encoder2(392,	98, false, Functions::LeakyRelu::f, Functions::LeakyRelu::df);
	Layer encoder3(98, 32, false, Functions::LeakyRelu::f, Functions::LeakyRelu::df);
	Layer middle(32, 98, false, Functions::LeakyRelu::f, Functions::LeakyRelu::df);
	Layer decoder1(98, 392, false, Functions::LeakyRelu::f, Functions::LeakyRelu::df);
	Layer decoder2(392, 784, false, Functions::Sigmoid::f, Functions::Sigmoid::df);
	Layer decoder3(784, 1, true);

	std::vector<Layer*> layers;
	layers.push_back(&encoder1);
	layers.push_back(&encoder2);
	layers.push_back(&encoder3);
	layers.push_back(&middle);
	layers.push_back(&decoder1);
	layers.push_back(&decoder2);
	layers.push_back(&decoder3);

	NeuralNetwork N(&layers, 0.1f, 0.3f, 1);
	Eigen::MatrixXf* buffer = new Eigen::MatrixXf(784, 1);

	Eigen::MatrixXf* res;// = N.predict_regress(buffer);

	Eigen::MatrixXf* dat = new Eigen::MatrixXf(1, 784);

	cimg_library::CImg<unsigned char> image;

	std::string userInput;
	int valBuffer;
	float fBuffer;

	size_t counter = 1;

	while (true)
	{
		std::cin >> userInput;
		
		if (userInput == "q")
			break;
		else if (userInput == "t")
		{
			std::cout << "Number of additionnal train : ";
			std::cin >> valBuffer;
			for (size_t i = 0; i < valBuffer; i++)
			{
				N.fit_regress(data, data);
			}
			std::cout << "Training is done (" << valBuffer << " steps)\n";
		}
		else if (userInput == "d")
		{
			valBuffer = rand() % nData;
			*buffer = data->block(valBuffer, 0, 1, 784).transpose();
			std::cout << "Displayed Number : " << A->operator()(valBuffer, 784) << std::endl;

			res = N.predict_regress(buffer);
			*dat = res->transpose();
			*dat = *dat * 255;

			write_matrix(*N.get_weights(2), "debug_weights" + std::to_string(counter) + ".txt");
			counter++;

			*dat = dat->unaryExpr(capping());
			image = data_to_img(dat, 28, 28);
			image.display();
		}
		else if (userInput == "w")
		{
			std::cout << "Index of weights to write : ";
			std::cin >> valBuffer;
			write_matrix(*N.get_weights(valBuffer), "debug_weights.txt");
		}
		else if (userInput == "l")
		{
			std::cout << "Current learning rate : " << N.get_learning_rate() << std::endl;
			std::cout << "New Value : ";
			std::cin >> fBuffer;
			N.set_learning_rate(fBuffer);
		}
		else if (userInput == "r")
		{
			std::cout << "Current regularization parameter : " << N.get_reg_param() << std::endl;
			std::cout << "New value : ";
			std::cin >> fBuffer;
			N.set_reg_param(fBuffer);
		}
		else
		{
			std::cout << "Wrong command\n";
		}
	}

	return 0;
}