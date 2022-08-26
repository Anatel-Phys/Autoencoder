#ifndef _FunctionsH
#define _FunctionsH
#include <cmath>
#include <fstream>
//commentary
#define LEAKY_RELU_SLOPE 0.3

static void write_matrix(Eigen::MatrixXf& mat, std::string fileName)
{
	std::ofstream of(fileName, 'w');
	for (size_t i = 0; i < mat.rows(); i++)
	{
		for (size_t j = 0; j < mat.cols() - 1; j++)
		{
			of << mat(i, j) << " ";
		}
		of << mat(i, mat.cols() - 1) << "\n";
	}
}

namespace Functions
{
	namespace Linear
	{
		inline float f(float x)
		{
			return x;
		}

		inline float df(float x)
		{
			return (float)1;
		}
	};

	namespace CappedRelu
	{
		inline float f(float x)
		{
			return std::fmax(std::fmax(0, x), std::fmin(x, 1));
		}

		inline float df(float x)
		{
			return (float)(x > 0 && x < 1);
		}
	}

	namespace Relu
	{
		inline float f(float x)
		{
			return std::fmax(0, x);
		}

		inline float df(float x)
		{
			return (float)(x > 0);
		}
	};

	namespace Sigmoid
	{
		inline float f(float x)
		{
			return (float)1 / ((float)1 + std::expf(-x));
		}

		inline float df(float x)
		{
			float sig = (float)1 / ((float)1 + std::expf(-x));
			return sig * ((float)1 - sig);
		}
	};

	namespace LeakyRelu
	{
		inline float f(float x)
		{
			if (x < 0)
				return LEAKY_RELU_SLOPE * x;
			return x;
		}
		
		inline float df(float x)
		{
			if (x < 0)
				return LEAKY_RELU_SLOPE;
			return 1;
		}
	}
	
	namespace Helper
	{
		inline Eigen::MatrixXf logic_vec(int classIdx, int nClasses)
		{
			Eigen::MatrixXf v = Eigen::MatrixXf::Zero(nClasses, 1);
			v(classIdx - 1, 0) = 1;
			return v;
		}

	};
};

#endif
