#pragma once

#include <assert.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <array>
#include <vector>
#include <map>

namespace ann {

	/**************************************************************************************************
	 * Random Generator
	 *
	 * Generate different random distributions intended for weights and bias initialization
	**************************************************************************************************/

	// Random Seed
	static int random_seed = 42;

	// Random Engine 
	static std::mt19937 gen(random_seed);

	// Set our generator with a random seed
	inline void set_random_seed(int seed)
	{
		random_seed = seed;
	}

	// Generate a random number under a normal distribution
	template<typename T>
	inline T get_random_normal(T mean = T(0), T stddev = T(1))
	{
		std::normal_distribution<T> dis(mean, stddev);
		return dis(gen);
	}

	// Generate a random number under a uniform distribution
	template<typename T>
	inline T get_random_uniform(T min = T(0), T max = T(1))
	{
		std::uniform_real_distribution<T> dis(min, max);
		return dis(gen);
	}

	// Generate a random number [0,1]
	template<typename T>
	inline T get_bernoulli_distribution(T probability = T(0.5))
	{
		std::bernoulli_distribution dis(probability);
		return dis(gen);
	}

	/**************************************************************************************************
	 * Prints a multidimensional tensor on console
	 *
	 * @param[in] rank - Number of dimensions. Ex: 1,2,3,4.
	 * @param[in] shapes - Number of elements inside each dimension. Ex: {2,3,4}.
	 * @param[in] data - Elements in vectorized format. Ex: {1,2,3,4,5,6,7...}.
	 * @param[in] offset and dimension_step are just for recursion purposes and should not be used.
	**************************************************************************************************/
	template<typename T>
	void print_tensor(std::ostream& os, const T * ptr, size_t rank, const size_t * shapes, size_t offset = 0, size_t dimension_step = 0)
	{
		if (dimension_step < rank - 1)
		{
			for (size_t i = 0; i < shapes[dimension_step]; i++)
			{
				print_tensor(os, ptr, rank, shapes, (i + offset)* shapes[dimension_step + 1], dimension_step + 1);
			}
			os << "\n";
		}
		else
		{
			for (size_t i = 0; i < shapes[dimension_step]; i++)
			{
				if (i != 0)
				{
					os << ", " << *(ptr + offset + i);
				}
				else {
					os << *(ptr + offset + i);
				}
			}
			os << "\n";
		}
	}

	/**
	 * Converts an image with dimensions (C,H,W) into columns with dimension (C * K_H * K_W, O_H * O_W)

	   This operation is mainly used for convolution layers to threat them as a simple matrix multiplication.

	   @Warning: The out_h and out_w paramters should be computed using the formula below!

	 * @param[in] image - Image in vectorized format.
	 * @param[in] c, in_h, in_w - channels, hight and width of input image
	 * @param[in] out_h = int((w + 2 * p_w - k_w - (k_w - 1) * (d_w - 1)) / s_w) + 1
	 * @param[in] out_w = int((h + 2 * p_h - k_h - (k_h - 1) * (d_h - 1)) / s_h) + 1
	 * @param[in] k_h, k_w - Kernel size
	 * @param[in] s_h, s_w - Stride size
	 * @param[in] p_h, p_w - Padding size
	 * @param[in] d_h, d_w - Dilation size
	 * @param[out] column - Image converted in column format [i_c * k_h * k_w, o_h * o_w]
	*/
	template<typename T>
	inline void im2col(
		const T * image,
		int c,
		int in_h, int in_w,
		int out_h, int out_w,
		int k_h, int k_w,
		int s_h, int s_w,
		int p_h, int p_w,
		int d_h, int d_w,
		T * column
	)
	{
		int idx_c, idx_w, idx_h, f_w, f_h, index_w, index_h, col_i, col_j, col_idx, img_idx, image_area, column_area;

		image_area = in_h * in_w;
		column_area = k_w * k_h * out_w * out_h;

		for (idx_c = 0; idx_c < c; idx_c++)
		{

			col_idx = 0;
			col_i = 0;
			for (idx_h = -p_h, index_h = 0; index_h < out_h; idx_h += s_h, index_h++)
			{
				for (idx_w = -p_w, index_w = 0; index_w < out_w; idx_w += s_w, index_w++)
				{
					col_j = 0;
					for (f_h = 0; f_h < k_h; f_h++)
					{

						for (f_w = 0; f_w < k_w; f_w++)
						{

							col_idx = col_j * out_h * out_w + col_i + column_area * idx_c;
							img_idx = (idx_h + d_h * f_h) * in_w + (idx_w + d_w * f_w) + image_area * idx_c;

							if (idx_w + d_w * f_w < 0 || idx_w + d_w * f_w >= in_w || idx_h + d_h * f_h < 0 || idx_h + d_h * f_h >= in_h)
							{
								column[col_idx] = 0;

							}
							else
							{
								column[col_idx] = image[img_idx];
							}
							col_idx++;
							col_j++;
						}

					}
					col_i++;
				}

			}
		}
	}

	/**
	 * Converts a column with dimensions (C * K_H * K_W, O_H * O_W) back as an image with dimensions (C,H,W)

	   This operation is mainly used for convolution layers to threat them as a simple matrix multiplication.

	   @Warning: The out_h and out_w paramters should be computed using the formula below!

	 * @param[in] column - Image converted in column format
	 * @param[in] c, in_h, in_w - channels, hight and width of input image
	 * @param[in] out_h = int((w + 2 * p_w - k_w - (k_w - 1) * (d_w - 1)) / s_w) + 1
	 * @param[in] out_w = int((h + 2 * p_h - k_h - (k_h - 1) * (d_h - 1)) / s_h) + 1
	 * @param[in] k_h, k_w - Kernel size
	 * @param[in] s_h, s_w - Stride size
	 * @param[in] p_h, p_w - Padding size
	 * @param[in] d_h, d_w - Dilation size
	 * @param[out] image - Image in vectorized format.
	*/
	template<typename T>
	inline void col2im(
		const T * column,
		int c,
		int in_h, int in_w,
		int out_h, int out_w,
		int k_h, int k_w,
		int s_h, int s_w,
		int p_h, int p_w,
		int d_h, int d_w,
		T * image
	)
	{
		int idx_c, idx_w, idx_h, f_w, f_h, index_w, index_h, col_i, col_j, col_idx, img_idx, image_area, column_area;

		image_area = in_h * in_w;
		column_area = k_w * k_h * out_w * out_h;

		// Reset values for image
		memset(image, 0, c * in_h * in_w * sizeof(float));

		for (idx_c = 0; idx_c < c; idx_c++)
		{
			col_idx = 0;
			col_i = 0;
			for (idx_h = -p_h, index_h = 0; index_h < out_h; idx_h += s_h, index_h++)
			{
				for (idx_w = -p_w, index_w = 0; index_w < out_w; idx_w += s_w, index_w++)
				{
					col_j = 0;
					for (f_h = 0; f_h < k_h; f_h++)
					{

						for (f_w = 0; f_w < k_w; f_w++)
						{

							col_idx = col_j * out_h * out_w + col_i + column_area * idx_c;
							img_idx = (idx_h + d_h * f_h) * in_w + (idx_w + d_w * f_w) + image_area * idx_c;

							if (idx_w + d_w * f_w >= 0 && idx_w + d_w * f_w < in_w && idx_h + d_h * f_h >= 0 && idx_h + d_h * f_h < in_h)
							{
								image[img_idx] += column[col_idx];

							}

							col_idx++;
							col_j++;
						}

					}
					col_i++;
				}
			}
		}
	}

	/**
	 * y = alpha * x + beta * y
	 *
	 * @param[in] n - Number of elements in input tensor.
	 * @param[in] alpha - Scalar alpha.
	 * @param[in] x - Input tensor.
	 * @param[in] inc_x - Storage spacing between elements of x.
	 * @param[in] beta - Scalar beta. Use 0.0 to set y or 1.0 to accumulate.
	 * @param[out] y - Output tensor.
	 * @param[in] inc_y - Storage spacing between elements of y.
	*/
	template<typename T>
	inline void axpby(
		const size_t n,
		const T alpha, const T *x, const int inc_x,
		const T beta, T *y, const int inc_y)
	{
		for (size_t i = 0; i < n; i++, x += inc_x, y += inc_y)
		{
			*y = alpha * *x + beta * *y;
		}
	}

	/**************************************************************************************************
	 * General Matrix Multiplication
	 *
	 * C = alpha * A * B + beta * C
	 *
	 * @param[in] transA - Specifies if matrix A is normal ('n') or transposed (any other char).
	 * @param[in] transB - Specifies if matrix B is normal ('n') or transposed (any other char).
	 * @param[in] m - Specifies the number of rows of matrix A and of the matrix C.
	 * @param[in] n - Specifies the number of columns of matrix B and of the matrix C.
	 * @param[in] k - Specifies the number of columns of matrix A and rows of the matrix B.
	 * @param[in] alpha - Scalar alpha.
	 * @param[in] A - Input matrix A.
	 * @param[in] lda - Specifies the first dimension of A. When transA = 'n' then
		   lda must be at least max( 1, m ), otherwise lda must be at least  max( 1, k ).
	 * @param[in] B - Input matrix B.
	 * @param[in] ldb - Specifies the first dimension of B. When transB = 'n' then
		   ldb must be at least max( 1, k ), otherwise ldb must be at least  max( 1, n ).
	 * @param[in] beta - Scalar beta. Use 0.0 to set y or 1.0 to accumulate.
	 * @param[out] C - Output matrix C.
	 * @param[in] ldc - Specifies the first dimension of C. Ldc must be at least max( 1, m ).
	**************************************************************************************************/
	template<typename T>
	inline void gemm(
		const char transA, const char transB,
		const int m, const int n, const int k,
		const T alpha,
		const T *A, const int lda,
		const T *B, const int ldb,
		const T beta,
		T * C, const int ldc)
	{
		int i, j, l, ncola, nrowa, nrowb;
		T sum;

		if (transA == 'n')
		{
			nrowa = m;
			ncola = k;
		}
		else
		{
			nrowa = k;
			ncola = m;
		}

		if (transB == 'n')
		{
			nrowb = k;
		}
		else
		{
			nrowb = n;
		}

		if (alpha == 0)
		{
			if (beta == 0)
			{
				for (j = 0; j < n; j++)
				{
					for (i = 0; i < m; i++)
					{
						C[i * ldc + j] = 0;
					}
				}
			}
			else
			{
				for (j = 0; j < n; j++)
				{
					for (i = 0; i < m; i++)
					{
						C[i * ldc + j] = beta * C[i * ldc + j];
					}
				}
			}
		}

		if (transB == 'n')
		{
			if (transA == 'n')
			{
				for (j = 0; j < n; j++)
				{
					if (beta == 0)
					{
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = 0;
						}
					}
					else if (beta != 1)
					{
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = beta * C[i * ldc + j];
						}
					}

					for (l = 0; l < k; l++)
					{
						sum = alpha * B[l * ldb + j];
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = C[i * ldc + j] + sum * A[i*lda + l];
						}
					}
				}
			}
			else
			{
				for (j = 0; j < n; j++)
				{
					for (i = 0; i < m; i++)
					{
						sum = 0;
						for (l = 0; l < k; l++)
						{
							sum = sum + A[l * lda + i] * B[l * ldb + j];
						}
						if (beta == 0)
						{
							C[i * ldc + j] = alpha * sum;
						}
						else
						{
							C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
						}
					}
				}
			}
		}
		else
		{
			if (transA == 'n')
			{
				for (j = 0; j < n; j++)
				{
					if (beta == 0)
					{
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = 0;
						}
					}
					else if (beta != 1)
					{
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = beta * C[i * ldc + j];
						}
					}

					for (l = 0; l < k; l++)
					{
						sum = alpha * B[j * ldb + l];
						for (i = 0; i < m; i++)
						{
							C[i * ldc + j] = C[i * ldc + j] + sum * A[i * lda + l];
						}
					}
				}
			}
			else
			{
				for (j = 0; j < n; j++)
				{
					for (i = 0; i < m; i++)
					{
						sum = 0;
						for (l = 0; l < k; l++)
						{
							sum = sum + A[l * lda + i] * B[j * ldb + l];
						}
						if (beta == 0)
						{
							C[i * ldc + j] = alpha * sum;
						}
						else
						{
							C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
						}
					}
				}
			}
		}
	}

	/**
	 * Mean over a multidimensional tensor
	 *
	 * mu = sum(x) / scale
	 *
	 * @param[in] x - Input tensor
	 * @param[in] n - Number of elements in input tensor.
	 * @param[in] shape - Number of elements in selected dimension.
	 * @param[in] stride - Number of elements to "skip" to get into the next element on this dimension.
	 * @param[out] mu - Output tensor containing mean over selected shape.
	*/
	template<typename T>
	inline void mean(
		const T * x,
		const size_t n,
		const size_t shape,
		const size_t stride,
		T * mu
	)
	{
		size_t i, j, k, slice, index_i;

		slice = size_t(n / (shape * stride));

		T scale = T(1) / (slice * stride);

		for (i = 0; i < shape; i++)
		{
			mu[i] = 0;
			for (j = 0; j < slice; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = j * stride * shape + i * stride + k;

					mu[i] += x[index_i];
				}
			}
			mu[i] *= scale;
		}

	}

	/**
	 * Variance over a multidimensional tensor
	 *
	 * mu = var(x) / scale
	 *
	 * @param[in] x - Input tensor
	 * @param[in] mu - Input tensor containing the mean of x.
	 * @param[in] n - Number of elements in input tensor.
	 * @param[in] shape - Number of elements in selected dimension.
	 * @param[in] stride - Number of elements to "skip" to get into the next element on this dimension.
	 * @param[out] var - Output tensor containing variance over selected shape.
	*/
	template<typename T>
	inline void variance(
		const T * x,
		const T * mu,
		const size_t n,
		const size_t shape,
		const size_t stride,
		T * var)
	{
		size_t i, j, k, slice, index_i;

		slice = int(n / (shape * stride));

		T scale = T(1) / (slice * stride);

		for (i = 0; i < shape; i++)
		{
			var[i] = 0;
			for (j = 0; j < slice; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = j * stride * shape + i * stride + k;

					var[i] += pow((x[index_i] - mu[i]), 2);
				}

			}
			var[i] *= scale;
		}
	}

	/**
	 * Normalization over a multidimensional tensor
	 *
	 * norm = (x - mean) / sqr(var)
	 *
	 * @param[in] x - Input tensor
	 * @param[in] mu - Input tensor containing the mean of x.
	 * @param[in] var - Input tensor containing the variance of x.
	 * @param[in] n - Number of elements in input tensor.
	 * @param[in] shape - Number of elements in selected dimension.
	 * @param[in] stride - Number of elements to "skip" to get into the next element on this dimension.
	 * @param[in] eps - Value added to denomination to maintain numerical stability
	 * @param[out] norm - Output tensor containing the normalized values over selected shape.
	*/
	template<typename T>
	inline void normalize(
		const T * x,
		const T * mu,
		const T * var,
		const size_t n,
		const size_t shape,
		const size_t stride,
		const T eps,
		T * norm)
	{
		size_t i, j, k, slice, index_i;

		slice = size_t(n / (shape * stride));

		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < shape; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = i * stride * shape + j * stride + k;

					norm[index_i] = (x[index_i] - mu[j]) / sqrt(var[j] + eps);
				}
			}
		}
	}

	/**
	 * Center data over a multidimensional tensor using gamma and beta
	 *
	 * y = norm * gamma + beta
	 *
	 * @param[in] norm - Normalized input tensor
	 * @param[in] gamma - Input tensor to scale the normalized tensor.
	 * @param[in] beta - Input tensor to offset the normalized tensor.
	 * @param[in] n - Number of elements in input tensor.
	 * @param[in] shape - Number of elements in selected dimension.
	 * @param[in] stride - Number of elements to "skip" to get into the next element on this dimension.
	 * @param[out] y - Output tensor containing the centered data over selected shape.
	*/
	template<typename T>
	inline void center(
		const T * norm,
		const T * gamma,
		const T * beta,
		const size_t n,
		const size_t shape,
		const size_t stride,
		T * y)
	{
		size_t i, j, k, slice, index_i;

		slice = size_t(n / (shape * stride));

		// y = gamma * x + beta
		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < shape; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = i * stride * shape + j * stride + k;

					y[index_i] = gamma[j] * norm[index_i] + beta[j];
				}
			}
		}
	}

	/**
	 * Add layer on feed forward operation
	 * y = x1 + alpha * x2
	 *
	 * @param[in] n - Number of elements by tensor.
	 * @param[in] alpha - Scaler for the second tensor
	 * @param[in] x1 - First input tensor
	 * @param[in] x2 - Second input tensor
	 * @param[out] y - Output tensor
	*/
	template<typename T>
	inline void add_layer_forward(const size_t n, const T alpha, const T * x1, const T * x2, T * y)
	{
		for (int i = 0; i < n; i++)
		{
			y[i] = x1[i] + alpha * x2[i];
		}
	}

	/**
	 * Add layer on backpropagation operation
	 * y = x1 + alpha * x2
	 *
	 * @param[in] n - Number of elements by tensor.
	 * @param[in] alpha - Scaler for the second tensor
	 * @param[in] dy - Output gradient tensor
	 * @param[out] dx1 - First gradient input tensor
	 * @param[out] dx2 - Second gradient input tensor
	*/
	template<typename T>
	inline void add_layer_backward(const size_t n, const T alpha, const T * dy, T * dx1, T *dx2)
	{
		for (int i = 0; i < n; i++)
		{
			dx1[i] = dy[i];
			dx2[i] = alpha * dy[i];
		}
	}

	/**************************************************************************************************
	 * Fully connected layer on feed forward operation
	 * y = x * w + b
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] n_inputs - Number of input neurons.
	 * @param[in] n_outputs - Number of output neurons.
	 * @param[in] x - Input tensor of shapes [batch, n_inputs]
	 * @param[in] w - Learnable weights of shapes [n_inputs, n_outputs]
	 * @param[in] b - Learnable bias of shapes [1, n_outputs]
	 * @param[out] y - Output tensor of shapes [batch, n_outputs]
	*/
	template<typename T>
	inline void fc_layer_forward(
		const size_t batch, const size_t n_inputs, const size_t n_outputs,
		const T * x,
		const T * w,
		const T * b,
		T * y
	)
	{
		int m, n, k;

		m = batch;
		n = n_outputs;
		k = n_inputs;

		// y[m,n] = x[m,k] * w[k,n]
		gemm<T>('n', 'n', m, n, k, 1, x, k, w, n, 0, y, n);

		// y[m,n] = y[m,n] + b[1,n]
		if (b)
		{
			for (int i = 0; i < m; i++)
			{
				axpby<T>(n, 1, b, 1, 1, y + i * n, 1);
			}
		}

	}

	/**************************************************************************************************
	 * Fully connected layer on backpropagation operation
	 * dw = x.T * dy
	 * db = sum(dy, axis=1)
	 * dx = dy * w.T
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] n_inputs - Number of input neurons.
	 * @param[in] n_outputs - Number of output neurons.
	 * @param[in] x - Input tensor of shapes [batch, n_inputs]
	 * @param[in] w - Learnable weights of size [n_inputs, n_outputs]
	 * @param[in] dy - Gradient of output tensor of shapes [batch, n_outputs]
	 * @param[out] dw - Gradient of learnable weights of shapes [n_inputs, n_outputs]
	 * @param[out] db - Gradient of learnable bias of shapes[1, n_outputs]
	 * @param[out] dx - Gradient of input tensor of shapes [batch, n_inputs]
	*/
	template<typename T>
	inline void fc_layer_backward(
		const size_t batch, const size_t n_inputs, const size_t n_outputs,
		const T * x,
		const T * w,
		const T * dy,
		T * dw, T * db, T * dx
	)
	{
		int m, n, k;

		m = n_inputs;
		n = n_outputs;
		k = batch;

		if (db)
		{
			for (int i = 0; i < k; i++)
			{
				axpby<T>(n, 1, dy + i * n, 1, 1, db, 1);
			}
		}

		// dw[m, n] = x[k, m].T * dy[k, n]
		gemm<T>('t', 'n', m, n, k, 1, x, m, dy, n, 1, dw, n);

		// dx[k, m] = dy[k, n] * w[m, n].T
		gemm<T>('n', 't', k, m, n, 1, dy, n, w, n, 1, dx, m);
	}

	/********************************************************************************************
	 * Convolution 2D layer on feed forward operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] in_c, in_h, in_w - Channels, height and width of input image.
	 * @param[in] out_c, out_h, out_w - Channels, height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, in_c, in_h, in_w]
	 * @param[in] w - 4D Tensor of learnable weights of shapes [out_c, in_c, k_h , k_w]
	 * @param[in] b - 1D Tensor of bias of shapes [out_c]
	 * @param[in] cols - 2D Tensor of size [in_c * k_h * k_w, out_h * out_w]
	 * @param[out] y - Output tensor of shapes [batch, out_c, out_h, out_w]

	 * out_h = int((in_h + 2 * p_h - k_h - (k_h - 1) * (d_h - 1)) / s_h) + 1;
	 * out_w = int((in_w + 2 * p_w - k_w - (k_w - 1) * (d_w - 1)) / s_w) + 1;

	**************************************************************************************************/
	template<typename T>
	inline void conv2d_layer_forward(
		const size_t batch,
		const size_t in_c, const size_t in_h, const size_t in_w,
		const size_t out_c, const size_t out_h, const size_t out_w,
		const size_t k_h, const size_t k_w,
		const size_t s_h, const size_t s_w,
		const size_t p_h, const size_t p_w,
		const size_t d_h, const size_t d_w,
		const T * x,
		const T * w,
		const T * b,
		T * cols,
		T * y)
	{
		size_t i, m, n, k, b_i, in_stride, out_stride;

		m = out_c;
		n = out_h * out_w;
		k = in_c * k_h * k_w;
		in_stride = in_c * in_h * in_w;
		out_stride = out_c * out_h * out_w;

		for (b_i = 0; b_i < batch; b_i++)
		{
			// Image to Colum conversion
			im2col(x + b_i * in_stride,
				in_c, in_h, in_w,
				out_h, out_w,
				k_h, k_w,
				s_h, s_w,
				p_h, p_w,
				d_h, d_w,
				cols);

			// Matrix multiplication 
			gemm<T>('n', 'n', m, n, k, 1, w, k, cols, n, 0, y + b_i * out_stride, n);

			// Add bias
			if (b)
			{
				for (i = 0; i < out_c; i++)
				{
					axpby<T>(n, T(1), b + i, 0, T(1), y + i * n + b_i * out_stride, 1);
				}
			}
		}
	}

	/********************************************************************************************
	 * Convolution 2D layer on backpropagation operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] in_c, in_h, in_w - Channels, height and width of input image.
	 * @param[in] out_c, out_h, out_w - Channels, height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, in_c, in_h, in_w]
	 * @param[in] w - 4D Tensor of learnable weights of shapes [out_c, in_c, k_h , k_w]
	 * @param[in] b - 1D Tensor of bias of shapes [out_c]
	 * @param[in] cols - 2D Tensor of size [in_c * k_h * k_w, out_h * out_w]
	 * @param[in] dy - Gradient of output tensor of shapes [batch, out_c, out_h, out_w]
	 * @param[out] dw - Gradient of learnable weights of shapes [out_c, in_c, k_h , k_w]
	 * @param[out] db - Gradient of learnable bias of shapes[out_c]
	 * @param[out] dx - Gradient of input tensor of shapes [batch, in_c, in_h, in_w]

	 * out_h = int((in_h + 2 * p_h - k_h - (k_h - 1) * (d_h - 1)) / s_h) + 1;
	 * out_w = int((in_w + 2 * p_w - k_w - (k_w - 1) * (d_w - 1)) / s_w) + 1;

	**************************************************************************************************/
	template<typename T>
	inline void conv2d_layer_backward(
		const size_t batch,
		const size_t in_c, const size_t in_h, const size_t in_w,
		const size_t out_c, const size_t out_h, const size_t out_w,
		const size_t k_h, const size_t k_w,
		const size_t s_h, const size_t s_w,
		const size_t p_h, const size_t p_w,
		const size_t d_h, const size_t d_w,
		const T * x,
		const T * w,
		const T * b,
		T * cols,
		const T * dy, T * dw, T * db, T * dx
	)
	{
		size_t m, n, k, b_i, in_stride, out_stride;

		m = out_c;
		n = in_c * k_h * k_w;
		k = out_h * out_w;
		in_stride = in_c * in_h * in_w;
		out_stride = out_c * out_h * out_w;

		if (db)
		{
			size_t  o_i;
			for (b_i = 0; b_i < batch; b_i++)
			{
				for (o_i = 0; o_i < out_c; o_i++)
				{
					axpby<T>(k, 1, dy + o_i * k + b_i * out_stride, 1, 1, db + o_i, 0);
				}
			}
		}

		for (b_i = 0; b_i < batch; b_i++)
		{
			// Image to Column conversion
			im2col<T>(x + b_i * in_stride,
				in_c, in_h, in_w,
				out_h, out_w,
				k_h, k_w,
				s_h, s_w,
				p_h, p_w,
				d_h, d_w,
				cols);

			// dw[k,n] = dot(dcosts[m,n], col_.T[n,k])
			gemm<T>('n', 't', m, n, k, 1, dy + b_i * out_stride, k, cols, k, 1, dw, n);

			// dx[m,k] = dot(w[m,n], dy[k,n].T)
			gemm<T>('t', 'n', n, k, m, 1, w, n, dy + b_i * out_stride, k, 0, cols, k);

			// Column to image
			col2im<T>(cols,
				in_c, in_h, in_w,
				out_h, out_w,
				k_h, k_w,
				s_h, s_w,
				p_h, p_w,
				d_h, d_w,
				dx + b_i * in_stride);
		}

	}

	/********************************************************************************************
	 * Convolution 2D transposed layer on feed forward operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] in_c, in_h, in_w - Channels, height and width of input image.
	 * @param[in] out_c, out_h, out_w - Channels, height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, in_c, in_h, in_w]
	 * @param[in] w - 4D Tensor of learnable weights of shapes [in_c, out_c, k_h, k_w]
	 * @param[in] b - 1D Tensor of bias of shapes [out_c]
	 * @param[in] cols - 2D Tensor of size [out_c * k_h * k_w, in_h * in_w]
	 * @param[out] y - Output tensor of shapes [batch, out_c, out_h, out_w]

	 * out_h = int((in_h - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + 1);
	 * out_w = int((in_w - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + 1);

	**************************************************************************************************/
	template<typename T>
	inline void conv2d_transposed_layer_forward(
		const size_t batch,
		const size_t in_c, const size_t in_h, const size_t in_w,
		const size_t out_c, const size_t out_h, const size_t out_w,
		const size_t k_h, const size_t k_w,
		const size_t s_h, const size_t s_w,
		const size_t p_h, const size_t p_w,
		const size_t d_h, const size_t d_w,
		const T * x,
		const T * w,
		const T * b,
		T * cols,
		T * y)
	{
		size_t i, b_i, m, n, k, in_stride, out_stride;

		m = out_c * k_h * k_w;
		n = in_h * in_w;
		k = in_c;
		in_stride = in_c * in_h * in_w;
		out_stride = out_c * out_h * out_w;

		for (b_i = 0; b_i < batch; b_i++)
		{
			// cols[out_c * k_h * k_w, in_h * in_w] = w[out_c * k_h * k_w, in_c].T * x[in_c , in_h * in_w]
			gemm<T>('t', 'n', m, n, k, 1, w, m, x + b_i * in_stride, n, 0, cols, n);

			// Column to image
			col2im(cols,
				out_c,
				out_h, out_w,
				in_h, in_w,
				k_h, k_w,
				s_h, s_w,
				p_h, p_w,
				d_h, d_w,
				y + b_i * out_stride);

			// Add bias
			if (b)
			{
				for (i = 0; i < out_c; i++)
				{
					axpby<T>(out_h * out_w, T(1), b + i, 0, T(1), y + i * n + b_i * out_stride, 1);
				}
			}
		}
	}

	/********************************************************************************************
	 * Deconvolution 2D layer on backpropagation operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] in_c, in_h, in_w - Channels, height and width of input image.
	 * @param[in] out_c, out_h, out_w - Channels, height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, in_c, in_h, in_w]
	 * @param[in] w - 4D Tensor of learnable weights of shapes [in_c, out_c, k_h, k_w]
	 * @param[in] b - 1D Tensor of bias of shapes [out_c]
	 * @param[in] cols - 2D Tensor of size [out_c * k_h * k_w * in_h * in_w]
	 * @param[in] dy - Gradient of output tensor of shapes [batch, out_c, out_h, out_w]
	 * @param[out] dw - Gradient of learnable weights of shapes [in_c, out_c, k_h, k_w]
	 * @param[out] db - Gradient of learnable bias of shapes[out_c]
	 * @param[out] dx - Gradient of input tensor of shapes [batch, in_c, in_h, in_w]

	 * out_h = int((in_h + 2 * p_h - k_h - (k_h - 1) * (d_h - 1)) / s_h) + 1;
	 * out_w = int((in_w + 2 * p_w - k_w - (k_w - 1) * (d_w - 1)) / s_w) + 1;

	**************************************************************************************************/
	template<typename T>
	inline void conv2d_transposed_layer_backward(
		const size_t batch,
		const size_t in_c, const size_t in_h, const size_t in_w,
		const size_t out_c, const size_t out_h, const size_t out_w,
		const size_t k_h, const size_t k_w,
		const size_t s_h, const size_t s_w,
		const size_t p_h, const size_t p_w,
		const size_t d_h, const size_t d_w,
		const T * x,
		const T * w,
		const T * b,
		T * cols,
		const T * dy, T * dw, T * db, T * dx
	)
	{
		size_t m, n, k, b_i, o_i, in_stride, out_stride;

		in_stride = in_c * in_h * in_w;
		out_stride = out_c * out_h * out_w;

		k = in_h * in_w;

		if (db)
		{
			for (b_i = 0; b_i < batch; b_i++)
			{
				for (o_i = 0; o_i < out_c; o_i++)
				{
					axpby<T>(out_h * out_w, 1, dy + o_i * k + b_i * out_stride, 1, 1, db + o_i, 0);
				}
			}
		}

		for (b_i = 0; b_i < batch; b_i++)
		{
			m = in_c;
			n = out_c * k_h * k_w;
			k = in_h * in_w;

			// dy[out_c * out_h * out_w] -> cols[out_c * k_h * k_w * in_h * in_w]
			im2col<T>(dy + b_i * out_stride,
				out_c,
				out_h, out_w,
				in_h, in_w,
				k_h, k_w,
				s_h, s_w,
				p_h, p_w,
				d_h, d_w,
				cols);

			// dw[in_c * out_c * k_h * k_w] = x[in_c , in_h * in_w] * cols[out_c * k_h * k_w , in_h * in_w].T
			gemm<T>('n', 't', m, n, k, 1, x + b_i * in_stride, k, cols, k, 1, dw, n);

			m = in_c;
			n = in_h * in_w;
			k = out_c * k_h * k_w;

			// dx[in_c, in_h * in_w] = w[in_c, out_c * k_h * k_w] * cols[out_c * k_h * k_w , in_h * in_w]
			gemm<T>('n', 'n', m, n, k, 1, w, k, cols, n, 1, dx + b_i * in_stride, n);

		}

	}

	/********************************************************************************************
	 * Max Pooling 2D layer on feed forward operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] channels - Number of input channels (same as output)
	 * @param[in] in_h, in_w - Height and width of input image.
	 * @param[in] out_h, out_w - Height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, channels, in_h, in_w]
	 * @param[in] indices - 1D Tensor of size [batch * channels * out_h * out_w]
	 * @param[out] y - 4D tensor with shapes [batch, channels, out_h, out_w]

	 * out_h = int(((in_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h) + 1);
	 * out_w = int(((in_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w) + 1);

	**************************************************************************************************/
	template<typename T>
	inline void maxpool2d_layer_forward(
		const size_t batch, const size_t channels,
		const size_t in_h, const size_t in_w,
		const size_t out_h, const size_t out_w,
		const size_t k_h, const size_t k_w,
		const size_t s_h, const size_t s_w,
		const size_t p_h, const size_t p_w,
		const size_t d_h, const size_t d_w,
		const T * x,
		T * indices,
		T * y)
	{
		int  pi_h, pi_w;

		T max_value;
		size_t max_index;

		size_t b_i, c_i, index_h, index_w, ki_h, ki_w, idx_input, idx_output;

		// Loop over batch
		for (b_i = 0; b_i < batch; b_i++)
		{
			// Loop over channel
			for (c_i = 0; c_i < channels; c_i++)
			{
				// Loop over height
				for (pi_h = -static_cast<int>(p_h), index_h = 0; index_h < out_h; pi_h += s_h, index_h++)
				{
					// Loop over width
					for (pi_w = -static_cast<int>(p_w), index_w = 0; index_w < out_w; pi_w += s_w, index_w++)
					{
						// Single Pooling
						max_value = std::numeric_limits<T>::lowest();

						for (ki_h = 0; ki_h < k_h; ki_h++)
						{
							for (ki_w = 0; ki_w < k_w; ki_w++)
							{
								idx_input = (d_h * ki_h + pi_h) * in_w +
									(d_w * ki_w + pi_w) +
									c_i * in_h * in_w +
									b_i * channels * in_h * in_w;

								if (d_h * ki_h + pi_h >= 0 &&
									d_h * ki_h + pi_h < in_h &&
									d_w * ki_w + pi_w >= 0 &&
									d_w * ki_w + pi_w < in_w)
								{
									if (x[idx_input] > max_value)
									{
										max_value = x[idx_input];
										max_index = idx_input;
									}
								}
							}
						}

						idx_output = index_w +
							index_h * out_w +
							c_i * out_h * out_w +
							b_i * channels * out_h * out_w;

						y[idx_output] = max_value;

						indices[idx_output] = static_cast<T>(max_index);
					}
				}
			}
		}
	}

	/********************************************************************************************
	 * Max Pooling 2D layer on backpropagation operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] channels - Number of input channels (same as output)
	 * @param[in] in_h, in_w - Height and width of input image.
	 * @param[in] out_h, out_w - Height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, channels, in_h, in_w]
	 * @param[in] indices - 1D Tensor of size [batch * channels * out_h * out_w]
	 * @param[in] dy - 4D tensor with shapes [batch, channels, out_h, out_w]
	 * @param[out] dx - 4D tensor with shapes [batch, channels, in_h, in_w]

	 * out_h = int(((in_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h) + 1);
	 * out_w = int(((in_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w) + 1);

	**************************************************************************************************/
	template<typename T>
	inline void maxpool2d_layer_backward(
		const size_t batch, const size_t channels,
		const size_t in_h, const size_t in_w,
		const size_t out_h, const size_t out_w,
		const size_t k_h, const size_t k_w,
		const size_t s_h, const size_t s_w,
		const size_t p_h, const size_t p_w,
		const size_t d_h, const size_t d_w,
		const T * indices,
		const T * dy,
		T * dx)
	{
		int  pi_h, pi_w;

		size_t b_i, c_i, index_h, index_w, ki_h, ki_w, idx_input, idx_filter;

		// Loop over batch
		for (b_i = 0; b_i < batch; b_i++)
		{
			// Loop over channel
			for (c_i = 0; c_i < channels; c_i++)
			{
				int index = 0;
				// Loop over height
				for (pi_h = -static_cast<int>(p_h), index_h = 0; index_h < out_h; pi_h += s_h, index_h++)
				{
					// Loop over width
					for (pi_w = -static_cast<int>(p_w), index_w = 0; index_w < out_w; pi_w += s_w, index_w++)
					{

						for (ki_h = 0; ki_h < k_h; ki_h++)
						{
							for (ki_w = 0; ki_w < k_w; ki_w++)
							{
								idx_input = (d_h * ki_h + pi_h) * in_w +
									(d_w * ki_w + pi_w) +
									c_i * in_h * in_w +
									b_i * channels * in_h * in_w;

								idx_filter = c_i * out_h * out_w +
									b_i * channels * out_h * out_w +
									index; // output area + output volume + index

								if (d_h *ki_h + pi_h >= 0 &&
									d_h *ki_h + pi_h < in_h &&
									d_w *ki_w + pi_w >= 0 &&
									d_w *ki_w + pi_w < in_w)
								{
									dx[idx_input] += (indices[idx_filter] == idx_input) * dy[idx_filter];
								}

							}
						}
						index++;
					}
				}
			}
		}
	}

	/**************************************************************************************************
	 * Zero Padding 2D layer on feed forward operation
	 *
	 * out_h = int(((in_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h) + 1);
	 * out_w = int(((in_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w) + 1);
	**************************************************************************************************/
	template<typename T>
	inline void zero_padding_2d_forward(
		const size_t batch, const size_t channels,
		const size_t in_h, const size_t in_w,
		const size_t out_h, const size_t out_w,
		const size_t padding_up, const size_t padding_left,
		const T * x, T * y)
	{
		size_t b_i, c_i, h_i, w_i;

		size_t index_x = 0;
		size_t index_y = 0;

		for (b_i = 0; b_i < batch; b_i++)
		{
			for (c_i = 0; c_i < channels; c_i++)
			{
				for (h_i = 0; h_i < in_h; h_i++)
				{
					for (w_i = 0; w_i < in_w; w_i++)
					{
						index_y = out_w * padding_up + padding_left +
							b_i * channels * out_h * out_w + // Volume
							c_i * out_h * out_w +
							w_i + h_i * out_w;

						y[index_y] = x[index_x];

						index_x++;
					}
				}
			}
		}
	}

	/**************************************************************************************************
	 * Zero Padding 2D layer on backward operation
	 *
	 * out_h = int(((in_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h) + 1);
	 * out_w = int(((in_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w) + 1);
	**************************************************************************************************/
	template<typename T>
	inline void zero_padding_2d_backward(
		const size_t batch, const size_t channels,
		const size_t in_h, const size_t in_w,
		const size_t out_h, const size_t out_w,
		const size_t padding_up, const size_t padding_left,
		const T * dy, T * dx)
	{
		size_t b_i, c_i, h_i, w_i;

		size_t index_x = 0;
		size_t index_y = 0;

		for (b_i = 0; b_i < batch; b_i++)
		{
			for (c_i = 0; c_i < channels; c_i++)
			{
				for (h_i = 0; h_i < in_h; h_i++)
				{
					for (w_i = 0; w_i < in_w; w_i++)
					{
						index_y = out_w * padding_up + padding_left +
							b_i * channels * out_h * out_w + // Volume
							c_i * out_h * out_w +
							w_i + h_i * out_w;

						dx[index_x] = dy[index_y];

						index_x++;
					}
				}
			}
		}
	}

	/********************************************************************************************
	 * Average Pooling 2D layer on feed forward operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] channels - Number of input channels (same as output)
	 * @param[in] in_h, in_w - Height and width of input image.
	 * @param[in] out_h, out_w - Height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, channels, in_h, in_w]
	 * @param[in] indices - 1D Tensor of size [batch * channels * out_h * out_w]
	 * @param[out] y - 4D tensor with shapes [batch, channels, out_h, out_w]

	 * out_h = int(((in_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h) + 1);
	 * out_w = int(((in_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w) + 1);

	**************************************************************************************************/
	template<typename T>
	inline void avgpool2d_layer_forward(
		const size_t batch, const size_t channels,
		const size_t in_h, const size_t in_w,
		const size_t out_h, const size_t out_w,
		const size_t k_h, const size_t k_w,
		const size_t s_h, const size_t s_w,
		const size_t p_h, const size_t p_w,
		const size_t d_h, const size_t d_w,
		const T * x,
		T * y)
	{
		int  pi_h, pi_w;

		T avg_value;

		size_t b_i, c_i, index_h, index_w, ki_h, ki_w, idx_input, idx_output;

		// Loop over batch
		for (b_i = 0; b_i < batch; b_i++)
		{
			// Loop over channel
			for (c_i = 0; c_i < channels; c_i++)
			{
				// Loop over height
				for (pi_h = -static_cast<int>(p_h), index_h = 0; index_h < out_h; pi_h += s_h, index_h++)
				{
					// Loop over width
					for (pi_w = -static_cast<int>(p_w), index_w = 0; index_w < out_w; pi_w += s_w, index_w++)
					{
						// Single Pooling
						avg_value = T(0);

						for (ki_h = 0; ki_h < k_h; ki_h++)
						{
							for (ki_w = 0; ki_w < k_w; ki_w++)
							{
								idx_input = (d_h * ki_h + pi_h) * in_w +
									(d_w * ki_w + pi_w) +
									c_i * in_h * in_w +
									b_i * channels * in_h * in_w;

								if (d_h * ki_h + pi_h >= 0 &&
									d_h * ki_h + pi_h < in_h &&
									d_w * ki_w + pi_w >= 0 &&
									d_w * ki_w + pi_w < in_w)
								{
									avg_value += x[idx_input];
								}
							}
						}

						idx_output = index_w +
							index_h * out_w +
							c_i * out_h * out_w +
							b_i * channels * out_h * out_w;

						y[idx_output] = avg_value / (ki_h * ki_w);
					}
				}
			}
		}
	}

	/********************************************************************************************
	 * Average Pooling 2D layer on backpropagation operation
	 *
	 * @param[in] batch - Number of input samples.
	 * @param[in] channels - Number of input channels (same as output)
	 * @param[in] in_h, in_w - Height and width of input image.
	 * @param[in] out_h, out_w - Height and width of output image.
	 * @param[in] k_h, k_w - Kernel height and width.
	 * @param[in] s_h, s_w - Stride height and width.
	 * @param[in] p_h, p_w - Padding height and width.
	 * @param[in] d_h, d_w - Dilation height and width.
	 * @param[in] x - 4D Tensor with shapes [batch, channels, in_h, in_w]
	 * @param[in] indices - 1D Tensor of size [batch * channels * out_h * out_w]
	 * @param[in] dy - 4D tensor with shapes [batch, channels, out_h, out_w]
	 * @param[out] dx - 4D tensor with shapes [batch, channels, in_h, in_w]

	 * out_h = int(((in_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h) + 1);
	 * out_w = int(((in_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w) + 1);

	**************************************************************************************************/
	template<typename T>
	inline void avgpool2d_layer_backward(
		const size_t batch, const size_t channels,
		const size_t in_h, const size_t in_w,
		const size_t out_h, const size_t out_w,
		const size_t k_h, const size_t k_w,
		const size_t s_h, const size_t s_w,
		const size_t p_h, const size_t p_w,
		const size_t d_h, const size_t d_w,
		const T * dy,
		T * dx)
	{
		int  pi_h, pi_w;

		size_t b_i, c_i, index_h, index_w, ki_h, ki_w, idx_input, idx_filter;

		// Loop over batch
		for (b_i = 0; b_i < batch; b_i++)
		{
			// Loop over channel
			for (c_i = 0; c_i < channels; c_i++)
			{
				int index = 0;
				// Loop over height
				for (pi_h = -static_cast<int>(p_h), index_h = 0; index_h < out_h; pi_h += s_h, index_h++)
				{
					// Loop over width
					for (pi_w = -static_cast<int>(p_w), index_w = 0; index_w < out_w; pi_w += s_w, index_w++)
					{

						for (ki_h = 0; ki_h < k_h; ki_h++)
						{
							for (ki_w = 0; ki_w < k_w; ki_w++)
							{
								idx_input = (d_h * ki_h + pi_h) * in_w +
									(d_w * ki_w + pi_w) +
									c_i * in_h * in_w +
									b_i * channels * in_h * in_w;

								idx_filter = c_i * out_h * out_w +
									b_i * channels * out_h * out_w +
									index; // output area + output volume + index

								if (d_h *ki_h + pi_h >= 0 &&
									d_h *ki_h + pi_h < in_h &&
									d_w *ki_w + pi_w >= 0 &&
									d_w *ki_w + pi_w < in_w)
								{
									dx[idx_input] += dy[idx_filter] / (k_h * k_w);
								}

							}
						}
						index++;
					}
				}
			}
		}
	}

	/********************************************************************************************
	 * Drop Out layer on feed forward operation
	 *
	 * @param[in] n - Number of elements.
	 * @param[in] prob - Probability threshold of setting neuron to zero (below this value).
	 * @param[in] x - Input tensor
	 * @param[out] y - Output tensor
	 * @param[out] mask - Probability mask for backpropagation
	**************************************************************************************************/
	template<typename T>
	inline void dropout_layer_forward(const size_t n, const T prob, const T * x, T * y, T * mask, bool is_training)
	{
		if (is_training)
		{
			T scale = T(1) / (T(1) - prob);

			for (size_t i = 0; i < n; i++)
			{
				float p = (float)rand() / RAND_MAX;
				mask[i] = p;

				if (p < prob) {
					y[i] = 0.0;
				}
				else
				{
					y[i] = x[i] * scale;
				}
			}
		}
		else
		{
			memcpy(y, x, sizeof(float)*n);
		}

	}

	/********************************************************************************************
	 * Drop Out layer on backpropagation operation
	 *
	 * @param[in] n - Number of elements.
	 * @param[in] prob - Probability threshold of setting neuron to zero (below this value).
	 * @param[in] dy - Gradient output
	 * @param[in] mask - Probability mask for backpropagation
	 * @param[out] dx - Gradient input
	**************************************************************************************************/
	template<typename T>
	inline void dropout_layer_backward(const size_t n, const T prob, const T * dy, const T * mask, T  * dx)
	{
		float scale = 1.0f / (1.0f - prob);

		for (int i = 0; i < n; i++)
		{
			float p = mask[i];
			if (p < prob) {
				dx[i] = 0.0f;
			}
			else
			{
				dx[i] = dy[i] * scale;
			}
		}
	}

	/********************************************************************************************
	 * Batch Normalization layer on feed forward operation
	 * Formula = (gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta)
	 *
	 * @param[in] n - Total number of elements from a tensor
	 * @param[in] shape - Shape along axis.
	 * @param[in] stride - Stride along axis.
	 * @param[in] momentum - Momentum for the moving average.
	 * @param[in] epsilon - Small float added to variance to avoid dividing by zero.
	 * @param[in] x - Input tensor. [d0, d1, ... dn]
	 * @param[in] mu - Mean of the input tensor. [shape over axis size]
	 * @param[in] var - Variance of the input tensor. [shape over axis size]
	 * @param[in] norm - Normalization of the input tensor calculated as (x - mu) / var. [d0, d1, ... dn]
	 * @param[in] gamma - Gamma weight. [shape over axis size]
	 * @param[in] beta - Beta weight. [shape over axis size]
	 * @param[in] moving_mean - Moving mean. [shape over axis size] - moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)
	 * @param[in] moving_variance - Moving variance. [shape over axis size] - moving_var = moving_var * momentum + var(batch) * (1 - momentum)
	 * @param[out] y - Output tensor [d0, d1, ... dn]
	 * @param[in] is_training - Flag indicating if we are going to normalize ou use the moving mean and average

	**************************************************************************************************/
	template<typename T>
	inline void batch_norm_forward(
		const size_t n,
		const size_t shape,
		const size_t stride,
		const T momentum,
		const T epsilon,
		const T * x,
		T * mu,
		T * var,
		T * norm,
		T * gamma,
		T * beta,
		T * moving_mean,
		T * moving_variance,
		T * y,
		bool is_training)
	{
		if (is_training)
		{
			// Calculat mean
			mean(x, n, shape, stride, mu);

			// Calculate variance
			variance(x, mu, n, shape, stride, var);

			// Calculate moving_mean = (1 - momentum) * mean + momentum * moving_mean  
			axpby(shape, T(1) - momentum, mu, 1, momentum, moving_mean, 1);

			// Calculate moving_var = (1 - momentum) * var + momentum * moving_var 
			axpby(shape, T(1) - momentum, var, 1, momentum, moving_variance, 1);

			// Normalize data
			normalize(x, mu, var, n, shape, stride, epsilon, norm);

			// Apply batch normalization
			center(norm, gamma, beta, n, shape, stride, y);
		}
		else
		{
			// Normalize data
			normalize(x, moving_mean, moving_variance, n, shape, stride, epsilon, norm);

			// Apply batch normalization
			center(norm, gamma, beta, n, shape, stride, y);
		}

	}

	/********************************************************************************************
	 * Batch Normalization layer on backpropagation operation
	 *
	 * @param[in] n - Total number of elements from a tensor
	 * @param[in] shape - Shape along axis.
	 * @param[in] stride - Stride along axis.
	 * @param[in] epsilon - Small float added to variance to avoid dividing by zero.
	 * @param[in] x - Input tensor. [d0, d1, ... dn]
	 * @param[in] mu - Mean of the input tensor. [shape over axis size]
	 * @param[in] var - Variance of the input tensor. [shape over axis size]
	 * @param[in] norm - Normalization of the input tensor calculated as (x - mu) / var. [d0, d1, ... dn]
	 * @param[in] gamma - Gamma weight. [shape over axis size]
	 * @param[in] beta - Beta weight. [shape over axis size]
	 * @param[in] dmu - Gradient of mean of the input tensor. [shape over axis size]
	 * @param[in] dvar - Gradient of variance of the input tensor. [shape over axis size]
	 * @param[out] dgamma - Gamma gradient. [shape over axis size]
	 * @param[out] dbeta - Beta gradient. [shape over axis size]
	 * @param[in] dy - Gradient output [d0, d1, ... dn]
	 * @param[out] dx - Gradient input [d0, d1, ... dn]

	**************************************************************************************************/
	template<typename T>
	inline void batch_norm_backward(
		const size_t n,
		const size_t shape,
		const size_t stride,
		const T epsilon,
		const T * x,
		const T * mu,
		const T * var,
		const T * norm,
		const T * gamma,
		const T * beta,
		T * dmu,
		T * dvar,
		T * dgamma,
		T * dbeta,
		const T * dy,
		T * dx
	)
	{
		size_t i, j, k, index_i;

		T std_inv = T(0);
		T x_mean = T(0);
		T dx_norm = T(0);
		T dx_mean = T(0);
		T dmean = T(0);
		T dvariance = T(0);
		T dg = T(0);
		T db = T(0);

		size_t N = n / shape;

		size_t slice = size_t(n / (shape * stride));

		// Calculate dgamma and dbeta
		for (i = 0; i < shape; i++)
		{
			dvariance = 0;
			dmean = 0;
			dg = 0;
			db = 0;
			dx_mean = 0;
			std_inv = T(1) / sqrt(var[i] + epsilon);

			for (j = 0; j < slice; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = j * stride * shape + i * stride + k;

					dg += dy[index_i] * norm[index_i];
					db += dy[index_i];

					x_mean = x[index_i] - mu[i];
					dx_norm = dy[index_i] * gamma[i];

					dvariance += dx_norm * x_mean * -T(0.5) * std_inv * std_inv * std_inv;
					dmean += dx_norm * -std_inv;
					dx_mean += -T(2) * x_mean / slice;
				}

			}

			dgamma[i] = dg;
			dbeta[i] = db;

			dvar[i] = dvariance;
			dmu[i] = dmean + dvariance * dx_mean;

		}

		// Calculate dx
		for (i = 0; i < shape; i++)
		{
			// std_inv = 1 / sqrt(var +eps)
			std_inv = T(1) / sqrt(var[i] + epsilon);

			for (j = 0; j < slice; j++)
			{
				for (k = 0; k < stride; k++)
				{
					index_i = j * stride * shape + i * stride + k;

					// dx_norm = dout * gamma
					dx_norm = dy[index_i] * gamma[i];

					// x_mean = x - mean
					x_mean = x[index_i] - mu[i];

					// dx = dx_norm * std_inv + dvar * 2 * x_mean / batch + dmean / batch
					dx[index_i] += dx_norm * std_inv + (dvar[i] * T(2) * x_mean) / N + dmu[i] / N;
				}
			}
		}
	}

	/**************************************************************************************************
	 * Sigmoid activation layer on feed forward operation
	 * y = 1/(1+exp(-x))
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[out] y - Output tensor in vectorized format
	**************************************************************************************************/
	template<typename T>
	inline void sigmoid_forward(const size_t n, const T * x, T * y)
	{
		for (size_t i = 0; i < n; i++, x++, y++)
		{
			*y = T(1) / (T(1) + exp(-*x));
		}
	}

	/**************************************************************************************************
	 * Sigmoid activation layer on backpropagation operation
	 * dx = 1/(1+exp(-x)) * (1 - 1/(1+exp(-x))) * dy
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	**************************************************************************************************/
	template<typename T>
	inline void sigmoid_backward(const size_t n, const T * x, const T * dy, T * dx)
	{
		for (size_t i = 0; i < n; i++, x++, dy++, dx++)
		{
			*dx = 1.0f / (1.0f + exp(-*x)) * (1.0f - 1.0f / (1.0f + exp(-*x))) * * dy;
		}
	}

	/**************************************************************************************************
	 * Fast sigmoid activation layer on backpropagation operation
	 * dx = y * (1 - y) * dy
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] y - Output tensor of sigmoid forward in vectorized format
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor of shapes [batch, n_inputs]
	**************************************************************************************************/
	template<typename T>
	inline void fast_sigmoid_backward(const size_t n, const T * y, const T * dy, T * dx)
	{
		for (int i = 0; i < n; i++, y++, dy++, dx++)
		{
			*dx = *y * (1.0f - *y) * * dy;
		}
	}

	/**************************************************************************************************
	 * Tanh activation layer on feed forward operation
	 * y = (e(x) - e(-x))/(e(x) + e(-x))
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[out] y - Output tensor in vectorized format
	**************************************************************************************************/
	template<typename T>
	inline void tanh_forward(const size_t n, const T * x, T * y)
	{
		for (size_t i = 0; i < n; i++, x++, y++)
		{
			*y = tanh(*x);
		}
	}

	/**************************************************************************************************
	 * Tanh activation layer on backpropagation operation
	 * dx = (1 - tanh(x)^2) * dy
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	**************************************************************************************************/
	template<typename T>
	inline void tanh_backward(const size_t n, const T * x, const T * dy, T * dx)
	{
		for (size_t i = 0; i < n; i++, x++, dy++, dx++)
		{
			*dx = (1.0f - tanh(*x) * tanh(*x)) * *dy;
		}
	}

	/**************************************************************************************************
	 * Fast tanh activation layer on backpropagation operation
	 * dx = (1 - y^2) * dy
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] y - Output tensor of tanh forward in vectorized format
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	**************************************************************************************************/
	template<typename T>
	inline void fast_tanh_backward(const size_t n, const T *y, const T *dy, T * dx)
	{
		for (int i = 0; i < n; i++, y++, dy++, dx++)
		{
			*dx = (T(1) - *y * *y) * *dy;
		}
	}

	/**************************************************************************************************
	 * ReLU activation layer on feed forward operation
	 * y = max(0, x)
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[out] y - Output tensor in vectorized format
	**************************************************************************************************/
	template<typename T>
	inline void relu_forward(const size_t n, const T * x, T * y)
	{
		for (size_t i = 0; i < n; i++, x++, y++)
		{
			*y = *x > 0.0f ? *x : 0.0f;
		}
	}

	/**************************************************************************************************
	 * ReLU activation layer on backpropagation operation
	 * dx = { x > 0 -> 1, x <= 0 -> 0} * dy
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] x - Input tensor in vectorized format
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	**************************************************************************************************/
	template<typename T>
	inline void relu_backward(const size_t n, const T * x, const T * dy, T * dx)
	{
		for (size_t i = 0; i < n; i++, x++, dy++, dx++)
		{
			*dx = (*x > 0.0f ? 1.0f : 0.0f) * *dy;
		}
	}

	/**************************************************************************************************
	 * Softmax activation layer on feed forward operation
	 * y = e(x)/ sum(e(x))
	 *
	 * @param[in] x - Input tensor in vectorized format
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] shape - Shape along axis.
	 * @param[in] stride - Stride along axis.
	 * @param[out] y - Output tensor in vectorized format
	**************************************************************************************************/
	template<typename T>
	inline void softmax_forward(const T * x, const size_t n, const size_t shape, const size_t stride, T * y)
	{
		size_t i, j, k, ind_x, slice;

		T max, sum;

		slice = size_t(n / (stride * shape));

		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < stride; j++)
			{
				max = -std::numeric_limits<T>::max();

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					max = (max > x[ind_x]) ? max : x[ind_x];
				}

				sum = T(0);

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					sum += exp(x[ind_x] - max);
				}

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					y[ind_x] = exp(x[ind_x] - max) / sum;
				}
			}
		}
	}

	/**************************************************************************************************
	 * Softmax activation layer on backpropagation operation
	 * dx = dy * jacobian matrix
	 *
	 * @param[in] y - Output tensor from softmax feed forward operation in vectorized format
	 * @param[in] n - Total number of elements from a tensor
	 * @param[in] shape - Shape along axis.
	 * @param[in] stride - Stride along axis.
	 * @param[in] dy - Gradient of output tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	**************************************************************************************************/
	template<typename T>
	inline void softmax_backward(const T * y, const size_t n, const size_t shape, const size_t stride, const T * dy, T * dx)
	{
		size_t i, j, k, l, ind_dx, ind_y, slice;

		T sum;

		slice = size_t(n / (stride * shape));

		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < stride; j++)
			{
				// Jacobian matrix
				for (k = 0; k < shape; k++)
				{
					sum = T(0);

					ind_dx = i * stride * shape + j + k * stride;

					for (l = 0; l < shape; l++)
					{
						ind_y = i * stride * shape + j + l * stride;

						sum += y[ind_dx] * (T(1) * (k == l) - y[ind_y]) * dy[ind_y];
					}

					dx[ind_dx] = sum;
				}
			}
		}
	}

	/**
	 * Mean Squared Error loss layer on feed forward operation
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format

	 * return loss = (input - target)^2
	*/
	template<typename T>
	inline T mse_loss_layer_forward(const size_t n, const T * input, const T * target)
	{
		T loss = 0.0f;

		for (size_t i = 0; i < n; i++)
		{
			loss += (input[i] - target[i]) * (input[i] - target[i]);
		}

		return loss / n;
	}

	/**
	 * Mean Squared Error loss layer on backpropagation operation
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format

	 * dx = 2 *(input - target) / n
	*/
	template<typename T>
	inline void mse_loss_layer_backward(const size_t n, const T * input, const T * target, T * dx)
	{
		for (size_t i = 0; i < n; i++)
		{
			dx[i] = 2.0f * (input[i] - target[i]) / n;
		}
	}

	/**
	 * Binary Cross Entropy loss layer on feed forward operation
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format

	 * return loss = -1 * (target * log(input) + (1-target) * log(1-input))
	 *
	 * @Warning: Input values must be between 0 and 1
	*/
	template<typename T>
	inline T binary_cross_entropy_loss_layer_forward(const size_t n, const T * input, const T * target)
	{
		float loss = 0.0f;

		for (int i = 0; i < n; i++)
		{
			loss += -1 * (target[i] * log(input[i] + std::numeric_limits<float>::epsilon()) +
				(1 - target[i]) * log(1 - input[i] + std::numeric_limits<float>::epsilon()));
		}

		return loss / n;
	}

	/**
	 * Binary Cross Entropy loss layer on backpropagation operation
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format
	 *
	 * dx = -1 * (target / input) + ((1-target)/(1-input))
	*/
	template<typename T>
	inline void binary_cross_entropy_loss_layer_backward(const size_t n, const T * input, const T * target, T * dx)
	{
		for (int i = 0; i < n; i++)
		{
			dx[i] = -1 * (target[i] / (input[i] + std::numeric_limits<float>::epsilon())) +
				(1 - target[i]) / (1 - input[i] + std::numeric_limits<float>::epsilon());

			dx[i] /= n;
		}
	}

	/**
	 * Huber loss layer on feed forward operation
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] delta - Threhold for smoothing
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format

	 * return loss = 0.5 * (input - target) ^2              -> if |input - target| < delta
			  loss = delta * (|input - target|- 0.5 * delta -> if |input - target| >= delta
	*/
	template<typename T>
	inline T huber_loss_layer_forward(const size_t n, const T delta, const T * input, const T * target)
	{
		T loss = T(0);

		for (size_t i = 0; i < n; i++)
		{
			if (abs(input[i] - target[i]) < delta)
			{
				loss += T(0.5) * (input[i] - target[i]) * (input[i] - target[i]);
			}
			else
			{
				loss += delta * (abs(input[i] - target[i]) - T(0.5) * delta);
			}
		}

		return loss / n;
	}

	/**
	 * Huber loss layer on backpropagation operation
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] delta - Threhold for smoothing
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format
	 * @param[out] dx - Gradient of input tensor in vectorized format

	 * dx = (input - target) / n    -> if |input - target| < delta
	 * dx = -/+ delta /n               -> if |input - target| >= delta
	*/
	template<typename T>
	inline void huber_loss_layer_backward(const size_t n, const T delta, const T * input, const T * target, T * dx)
	{
		for (size_t i = 0; i < n; i++)
		{
			if (abs(input[i] - target[i]) < delta)
			{
				dx[i] = input[i] - target[i];
			}
			else if (input[i] - target[i] < T(0))
			{
				dx[i] = -delta;
			}
			else
			{
				dx[i] = delta;
			}

			dx[i] /= n;
		}
	}

	/**
	 * Cross Entropy loss layer on feed forward operation
	 *
	 * @param[in] n - Input tensor in vectorized format
	 * @param[in] shapes - Number of elements inside each dimension. Ex: {2,3,4}.
	 * @param[in] axis - Axis along which the softmax operation will be performed.
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format
	*/
	template<typename T>
	inline T cross_entropy_loss_layer_forward(
		const size_t n,
		const size_t shape,
		const size_t stride,
		const T * input,
		const T * target
	)
	{
		size_t i, j, k, ind_x, slice;

		T sum_input, error;

		T loss = T(0);

		slice = int(n / (stride * shape));

		T eps = T(1e-7);

		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < stride; j++)
			{
				// Input normalization
				sum_input = eps;

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					sum_input += input[ind_x];
				}

				error = T(0);

				// Cross Entropy error
				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					// Clip input between 1-eps and eps before taking the log
					error += -log(std::max(std::min(input[ind_x] / sum_input, T(1) - eps), eps)) * target[ind_x];
				}

				loss += error;
			}
		}

		loss /= n / shape;

		return loss;

	}

	/**
	 * Cross Entropy loss layer on backpropagation operation
	 *
	 * @param[in] n - Input tensor in vectorized format
	 * @param[in] shapes - Number of elements inside each dimension. Ex: {2,3,4}.
	 * @param[in] axis - Axis along which the softmax operation will be performed.
	 * @param[in] input - Input tensor in vectorized format
	 * @param[in] target - Target tensor in vectorized format
	*/
	template<typename T>
	inline void cross_entropy_loss_layer_backward(
		const size_t n,
		const size_t shape,
		const size_t stride,
		const T * input,
		const T * target,
		T * dx
	)
	{
		size_t i, j, k, ind_x, slice;

		T sum_target, sum_input;

		slice = int(n / (stride * shape));

		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < stride; j++)
			{

				// Input / Target normalization
				sum_target = 0.0f;
				sum_input = 0.0f;

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					sum_input += input[ind_x];
					sum_target += target[ind_x];
				}

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					dx[ind_x] = sum_target / sum_input - (target[ind_x] / input[ind_x]);

					dx[ind_x] /= n / shape;

				}

			}
		}
	}

	template<typename T>
	inline T cross_entropy_loss_from_logits_layer_forward(
		const size_t n,
		const size_t shape,
		const size_t stride,
		const T * input,
		const T * target
	)
	{
		size_t i, j, k, ind_x, slice;

		T max, sum, error;

		T loss = T(0);

		slice = int(n / (stride * shape));

		T eps = T(1e-7f);

		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < stride; j++)
			{

				error = 0.0f;

				max = -std::numeric_limits<T>::max();

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					max = (max > input[ind_x]) ? max : input[ind_x];
				}

				sum = 0.0f;

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					sum += exp(input[ind_x] - max);
				}

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					error += -log(exp(input[ind_x] - max) / sum) * target[ind_x];
				}


				loss += error;
			}
		}

		loss /= n / shape;

		return loss;
	}

	template<typename T>
	inline void cross_entropy_loss_from_logits_layer_backward(
		const size_t n,
		const size_t shape,
		const size_t stride,
		const T * input,
		const T * target,
		T * dx
	)
	{
		size_t i, j, k, ind_x, slice;

		T max, sum;

		slice = int(n / (stride * shape));

		T eps = T(1e-7f);

		for (i = 0; i < slice; i++)
		{
			for (j = 0; j < stride; j++)
			{

				max = -std::numeric_limits<float>::max();

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					max = (max > input[ind_x]) ? max : input[ind_x];
				}

				sum = 0.0f;

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					sum += exp(input[ind_x] - max);
				}

				for (k = 0; k < shape; k++)
				{
					ind_x = i * stride * shape + j + k * stride;

					dx[ind_x] = exp(input[ind_x] - max) / sum - target[ind_x];

					dx[ind_x] /= n / shape;

				}

			}
		}
	}

	/**
	 * Stochastic Gradient Descent Optimizer
	 * w = w - learning rate * g
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] lr - Learning rate
	 * @param[in] mom - Momentum vector
	 * @param[in] g - Gradient vector
	 * @param[in] v - Velocity vector
	 * @param[out] w - Weight vector
	 *
	*/
	template<typename T>
	inline void sgd_optimizer(
		const size_t n, const T lr, const T mom,
		const T * g, T * v, T * w
	)
	{
		for (size_t i = 0; i < n; i++, g++, v++, w++)
		{
			if (mom > T(0))
			{
				*v = mom * *v + *g;
				*w = *w - lr * *v;
			}
			else
			{
				*w = *w - lr * *g;
			}

		}
	}

	/**
	 * Root Mean Squared Optimizer
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] lr - Learning rate
	 * @param[in] rho - Discounting factor for the history gradient
	 * @param[in] mom - Momentum vector
	 * @param[in] eps - Constant for numerical stability.
	 * @param[in] g - Gradient vector
	 * @param[in] v - Velocity vector
	 * @param[out] w - Weight vector
	 *
	*/
	template<typename T>
	inline void rms_prop_optimizer(
		const size_t n, const T lr, const T rho, const T mom, const T eps,
		const T * g, T * v, T * w
	)
	{
		for (size_t i = 0; i < n; i++, g++, v++, w++)
		{
			*v = rho * *v + (T(1) - rho) * *g * *g;

			*w = *w - lr * *g / (sqrt(*v) + eps);
		}
	}

	/**
	 * Adaptive Moment Estimation Optimizer
	 *
	 * @param[in] n - Number of elements to perform the operation.
	 * @param[in] iter - Iteration number.
	 * @param[in] lr - Learning rate
	 * @param[in] beta_1 - Exponential decay rate for the 1st moment estimates.
	 * @param[in] beta_2 - Exponential decay rate for the 2nd moment estimates.
	 * @param[in] eps - Constant for numerical stability.
	 * @param[in] mom - Momentum vector
	 * @param[in] g - Gradient vector
	 * @param[in] v - Velocity vector
	 * @param[in] m - Momentum vector
	 * @param[out] w - Weight vector
	 *
	*/
	template<typename T>
	inline void adam_optimizer(
		const size_t n, const size_t iter, const T lr, const T beta_1, const T beta_2, const T eps,
		const T * g, T * v, T * m, T * w
	)
	{
		T m_corr, v_corr;

		for (size_t i = 0; i < n; i++, g++, v++, m++, w++)
		{

			*m = beta_1 * *m + (T(1) - beta_1) * *g;

			*v = beta_2 * *v + (T(1) - beta_2) * *g * *g;

			m_corr = *m / (T(1) - pow(beta_1, float(iter)));

			v_corr = *v / (T(1) - pow(beta_2, float(iter)));

			*w = *w - m_corr * lr / (sqrt(v_corr) + eps);
		}
	}

	template<size_t N>
	class IntList
	{
	public:

		IntList(size_t value)
		{
			for (size_t i = 0; i < N; i++)
			{
				data_[i] = value;
			}
		}

		IntList& operator=(size_t value)
		{
			for (size_t i = 0; i < N; i++)
			{
				data_[i] = value;
			}

			return *this;
		}

		IntList(const std::initializer_list<size_t>& data)
		{
			assert(data.size() == N);

			std::copy(data.begin(), data.end(), data_.begin());
		}

		IntList& operator=(const std::initializer_list<size_t>& data)
		{
			assert(data.size() == N);

			for (size_t i = 0; i < N; i++)
			{
				data_[i] = data[i];
			}

			return *this;
		}

		size_t operator[](size_t index) { return data_[index]; }

	private:
		std::array<size_t, N> data_;
	};


	template<typename T>
	class Tensor
	{
	public:

		Tensor()
		{

		};

		~Tensor()
		{
			clear();
		};

		explicit Tensor(const std::vector<size_t>& shapes)
		{
			reshape(shapes);
		};

		Tensor(const Tensor& other)
		{
			reshape(other.shapes());

			set_data(other.size(), other.data());

			set_grad(other.size(), other.grad());
		}

		Tensor& operator=(const Tensor& other)
		{
			if (this != &other)
			{
				reshape(other.shapes());

				set_data(other.size(), other.data());

				set_grad(other.size(), other.grad());
			}

			return *this;

		}

		Tensor(Tensor&& other) noexcept :
			shapes_(other.shapes_),
			strides_(other.strides_),
			size_(other.size_),
			capacity_(other.capacity_)
		{

			data_ = other.data();
			grad_ = other.grad();

			other.data_ = nullptr;
			other.grad_ = nullptr;
		}

		Tensor& operator=(const Tensor&& other)
		{
			if (this != &other)
			{
				shapes_ = other.shapes_;
				strides_ = other.strides_;
				size_ = other.size_;
				capacity_ = other.capacity_;

				data_ = other.data();
				grad_ = other.grad();

				other.data_ = nullptr;
				other.grad_ = nullptr;
			}

			return *this;
		}

		// Getters
		T* data() { return data_; }
		const T* data() const { return data_; }

		T* grad() { return grad_; }
		const T* grad() const { return grad_; }

		const size_t rank() const { return shapes_.size(); }
		const size_t size() const { return size_; }

		const std::vector<size_t>& shapes() const { return shapes_; }
		const std::vector<size_t>& strides() const { return strides_; }

		const size_t shape(int index)  const { return index < 0 ? shapes_[shapes_.size() + index] : shapes_[index]; }
		const size_t stride(int index)  const { return index < 0 ? strides_[strides_.size() + index] : strides_[index]; }

		// Setters

		void set_data(const std::vector<T>& values, size_t offset = 0)
		{
			set_data(values.size(), values.data(), offset);
		}

		void set_data(size_t size, const T * values, size_t offset = 0)
		{
			size_t count = 0;

			for (size_t i = offset; i < size + offset; i++)
			{
				data_[i] = values[count];

				count++;
			}
		}

		void set_grad(const std::vector<T>& values, size_t offset = 0)
		{
			set_grad(values.size(), values.data(), offset);
		}

		void set_grad(size_t size, const T * values, size_t offset = 0)
		{
			size_t count = 0;

			for (size_t i = offset; i < size + offset; i++)
			{
				grad_[i] = values[count];

				count++;
			}
		}

		// Tensor subscription

		T& operator[](size_t index) { return data_[index]; }
		const T& operator[](size_t index) const { return data_[index]; }

		T& operator()(const std::vector<size_t>& indices) { return *(data_ + offset(indices)); }
		const T& operator()(const std::vector<size_t>& indices) const { return *(data_ + offset(indices)); }

		T& data_at(size_t index) { return data_[index]; }
		const T& data_at(int index) const { return data_[index]; }

		T& grad_at(size_t index) { return grad_[index]; }
		const T& grad_at(int index) const { return grad_[index]; }

		void reshape(const std::vector<size_t>& shapes)
		{

			// Don't do anything if shapes are equal
			if (shapes == shapes_)
				return;

			// Copy new shapes
			shapes_ = shapes;

			// Calculate number of elements according to the shape provided
			size_ = calculate_size(shapes);

			// Resize if more elements are needed
			if (size_ > capacity_)
			{
				capacity_ = size_;

				resize(size_);
			}

			//Calculate stride
			calculate_strides();

		};

		// Check if tensor is not initialized
		const bool empty() const { return size_ == 0; }

		void clear()
		{
			// Clear all parameters
			size_ = 0;
			capacity_ = 0;
			shapes_.clear();

			// Clear data and gradients
			delete[] data_;
			delete[] grad_;
		}

	private:

		void resize(size_t size)
		{

			T * new_data_ = new T[size]();
			T * new_grad_ = new T[size]();

			delete[] data_;
			delete[] grad_;

			data_ = new_data_;
			grad_ = new_grad_;
		}

		void calculate_strides()
		{

			if (strides_.size() != shapes_.size())
				strides_.resize(shapes_.size());

			strides_[shapes_.size() - 1] = 1;
			for (size_t i = shapes_.size() - 1; i != 0; --i)
			{
				strides_[i - 1] = strides_[i] * shapes_[i];
			}
		}

		// Calculate size of the flat tensor based on a vector list
		size_t calculate_size(const std::vector<size_t>& shapes)
		{
			size_t total_size = 1;

			for (size_t i = 0; i < shapes.size(); i++)
			{
				total_size *= shapes[i];
			}

			return total_size;
		}

		size_t offset(const std::vector<size_t>& indices)
		{

			size_t index = 0;

			for (size_t i = 0; i < strides_.size(); i++)
			{
				index += strides_[i] * indices[i];
			}

			return index;
		}



	private:

		size_t size_{ 0 };
		size_t capacity_{ 0 };

		T * data_{ nullptr };
		T * grad_{ nullptr };

		std::vector<size_t> shapes_;
		std::vector<size_t> strides_;

	};

	// Print tensor
	template<typename T>
	inline std::ostream& operator << (std::ostream& os, const Tensor<T>& tensor)
	{

		if (tensor.empty())
		{
			os << "Tensor not initialized" << "\n";

			return os;
		}

		os << "Tensor shapes=(";

		const std::vector<size_t>& shapes = tensor.shapes();

		for (size_t i = 0; i < shapes.size(); i++)
		{

			if (i != 0)
			{
				os << shapes[i];
			}
			else {
				os << shapes[i] << ",";
			}
		}

		os << ")\n";

		print_tensor(os, tensor.data(), tensor.rank(), tensor.shapes().data());

		print_tensor(os, tensor.grad(), tensor.rank(), tensor.shapes().data());

		return os;
	}

	/********************************************************************
	 Layers
	*********************************************************************/
	template<typename T>
	struct LayerConfig
	{
		size_t n_features{ 0 };
		size_t n_inputs{ 0 };
		size_t n_outputs{ 0 };

		size_t in_c{ 1 };
		size_t in_h{ 1 };
		size_t in_w{ 1 };

		size_t out_c{ 1 };
		size_t out_h{ 1 };
		size_t out_w{ 1 };

		size_t k_c{ 2 };
		size_t k_h{ 2 };
		size_t k_w{ 2 };

		size_t s_c{ 1 };
		size_t s_h{ 1 };
		size_t s_w{ 1 };

		size_t p_c{ 0 };
		size_t p_h{ 0 };
		size_t p_w{ 0 };

		size_t p_u{ 0 };
		size_t p_l{ 0 };
		size_t p_d{ 0 };
		size_t p_r{ 0 };

		size_t d_c{ 1 };
		size_t d_h{ 1 };
		size_t d_w{ 1 };

		int axis{ -1 };

		bool has_bias{ true };
		bool is_training{ false };

		T momentum{ 0.99 };
		T epsilon{ 0.001 };
		T alpha{ 0.5 };

	};

	template<typename T>
	class ILayer
	{
	public:

		//Get layer type
		virtual const std::string type() const = 0;

		// Get number of output tensors
		virtual size_t get_output_count() { return 1; }

		// Feed Forward. Do NOT override this method
		void forward(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs)
		{

			reshape(inputs, outputs);

			forward_impl(inputs, outputs);

		}

		// Backpropagation. Do NOT override this method
		void backward(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs)
		{
			backward_impl(inputs, outputs);
		}

		// Get learnable parameters
		std::vector<Tensor<T>*>& parameters() { return parameters_; }

		virtual void set_parameters(size_t index, const std::vector<T>& parameters)
		{
			if (parameters.size() != parameters_[index]->size())
			{
				throw "Number of parameters";
			}

			T * p = parameters_[index]->data();

			for (size_t i = 0; i < parameters_[index]->size(); i++, p++)
			{
				*p = parameters[i];
			}
		}

		// Get total number for parameters
		size_t get_total_parameters()
		{
			size_t n_parameters = 0;

			for (size_t i = 0; i < parameters_.size(); i++)
			{
				n_parameters += parameters_[i]->size();
			}

			return n_parameters;
		}

	protected:

		// Reshape 
		virtual void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) = 0;

		// Feed Forward 
		virtual void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) = 0;

		// Backpropagation
		virtual void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) = 0;

	protected:

		std::vector<Tensor<T>*> parameters_;

		LayerConfig<T> config_;

	};

	template<typename T>
	class FCLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "fc_layer"; };

		FCLayer(size_t n_inputs, size_t n_outputs, bool has_bias)
		{
			this->config_.n_inputs = n_inputs;
			this->config_.n_outputs = n_outputs;
			this->config_.has_bias = has_bias;

			build();
		}

		void build()
		{
			int n_paramters = this->config_.has_bias ? 2 : 1;
			this->parameters_.resize(n_paramters);

			// Create weights
			this->parameters_[0] = new Tensor<T>({ this->config_.n_inputs , this->config_.n_outputs });

			// Initialize Weights
			T k = T(1) / this->config_.n_inputs;

			auto w = this->parameters_[0]->data();

			for (size_t i = 0; i < this->parameters_[0]->size(); i++, w++)
			{
				*w = get_random_uniform<T>(-sqrt(k), sqrt(k));
			}

			// Initialize bias
			if (this->config_.has_bias)
			{
				this->parameters_[1] = new Tensor<T>({ 1 , this->config_.n_outputs });
			}
		}

		void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
				- Input single tensor rank > 1
			*/

			assert(inputs.size() == 1 && outputs.size() == 1);

			assert(inputs[0]->rank() > 1);

			assert(inputs[0]->stride(0) > this->config_.n_outputs);

			// Input Shape (N, *, n_inputs)
			// Output Shape (N, *, n_outputs)
			outputs[0]->reshape({ inputs[0]->shape(0) , this->config_.n_outputs });
		}

		void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{

			T * x = inputs[0]->data();
			T * y = outputs[0]->data();
			T * b = this->parameters_[1]->data();//b[1,n]
			T * w = this->parameters_[0]->data(); //W[k,n]

			fc_layer_forward(
				outputs[0]->shape(0),
				this->config_.n_inputs,
				this->config_.n_outputs,
				x, w, b, y);
		}

		void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{

			T * w = this->parameters_[0]->data(); //[n_inputs, n_outputs]
			T * x = inputs[0]->data(); //[batch, n_inputs]

			// Grads
			T * dw = this->parameters_[0]->grad(); //[n_inputs, n_outputs]
			T * db = this->parameters_[1]->grad(); // [1, n_outputs]
			T * dx = inputs[0]->grad(); // [batch, n_inputs]
			T * dy = outputs[0]->grad(); // [batch, n_outputs]

			fc_layer_backward(
				inputs[0]->shape(0),
				this->config_.n_inputs,
				this->config_.n_outputs,
				x, w, dy, dw, db, dx
			);

		}

	};

	template<typename T>
	class ZeroPadding2DLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "zero_padding_2d_layer"; };

		ZeroPadding2DLayer(IntList<2> padding = 1)
		{
			this->config_.p_u = padding[0];
			this->config_.p_d = padding[0];
			this->config_.p_l = padding[1];
			this->config_.p_r = padding[1];
		};

		ZeroPadding2DLayer(size_t padding_up, size_t padding_down, size_t padding_left, size_t padding_right)
		{
			this->config_.p_u = padding_up;
			this->config_.p_d = padding_down;
			this->config_.p_l = padding_left;
			this->config_.p_r = padding_right;
		};

		~ZeroPadding2DLayer() {};

		// Reshape
		virtual void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
				- Rank = 4
			*/
			assert(inputs.size() == 1 && outputs.size() == 1);

			// Format BCHW - Batch, Channels, Height, Width
			assert(inputs[0]->rank() == 4);

			config_.in_h = inputs[0]->shape(2);

			config_.in_w = inputs[0]->shape(3);

			size_t out_h = inputs[0]->shape(2) + this->config_.p_u + this->config_.p_d;

			size_t out_w = inputs[0]->shape(3) + this->config_.p_l + this->config_.p_r;

			outputs[0]->reshape({ inputs[0]->shape(0), inputs[0]->shape(1), out_h, out_w });

		};

		// Feed Forward
		virtual void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * y = outputs[0]->data();

			zero_padding_2d_forward<T>(
				inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3),
				inputs[0]->shape(2) + this->config_.p_u + this->config_.p_d,
				inputs[0]->shape(3) + this->config_.p_l + this->config_.p_r,
				this->config_.p_u, this->config_.p_l,
				x, y);
		};

		// Backward
		virtual void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * dx = inputs[0]->grad();
			T * dy = outputs[0]->grad();

			zero_padding_2d_backward(
				inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3),
				inputs[0]->shape(2) + this->config_.p_u + this->config_.p_d,
				inputs[0]->shape(3) + this->config_.p_l + this->config_.p_r,
				this->config_.p_u, this->config_.p_l,
				dy, dx);
		};

	};

	template<typename T>
	class Conv2DLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "conv_2d_layer"; };

		explicit Conv2DLayer(
			int in_channels, int out_channels,
			IntList<2> kernel,
			IntList<2> stride = 1,
			IntList<2> padding = 0,
			IntList<2> dilation = 1,
			bool has_bias = true
		)
		{
			this->config_.in_c = in_channels;
			this->config_.out_c = out_channels;

			this->config_.k_h = kernel[0];
			this->config_.k_w = kernel[1];
			this->config_.s_h = stride[0];
			this->config_.s_w = stride[1];
			this->config_.p_h = padding[0];
			this->config_.p_w = padding[1];
			this->config_.d_h = dilation[0];
			this->config_.d_w = dilation[1];

			this->config_.has_bias = has_bias;

			build();
		}

		Conv2DLayer(
			int in_channels, int out_channels,
			int kernel_height, int kernel_width,
			int stride_height = 1, int stride_width = 1,
			int padding_height = 0, int padding_width = 0,
			int dilation_height = 1, int dilation_width = 1,
			bool has_bias = true)
		{
			this->config_.in_c = in_channels;
			this->config_.out_c = out_channels;

			this->config_.k_h = kernel_height;
			this->config_.k_w = kernel_width;
			this->config_.s_h = stride_height;
			this->config_.s_w = stride_width;
			this->config_.p_h = padding_height;
			this->config_.p_w = padding_width;
			this->config_.d_h = dilation_height;
			this->config_.d_w = dilation_width;

			this->config_.has_bias = has_bias;

			build();
		}

		// Layer Setup
		void build()
		{
			// Define number of parameters that will be stores (weights and/or bias)
			int n_paramters = this->config_.has_bias ? 2 : 1;
			this->parameters_.resize(n_paramters);

			// weight - 4D Tensor of learnable weights of shapes[out_c, in_c, k_h, k_w]
			this->parameters_[0] = new Tensor<T>({
				this->config_.out_c,  // Output channels
				this->config_.in_c,   // Input channels 
				this->config_.k_h,    // Kernel height
				this->config_.k_w     // Kernel width
				});

			// Initialize Weights
			T k = T(1) / (this->config_.in_c * this->config_.k_h * this->config_.k_w);

			auto w = this->parameters_[0]->data();

			for (size_t i = 0; i < this->parameters_[0]->size(); i++, w++)
			{
				*w = get_random_uniform<T>(-sqrt(k), sqrt(k));
			}

			// bias - 1D Tensor of bias of shapes[out_c]
			if (this->config_.has_bias)
			{
				this->parameters_[1] = new Tensor<T>({ this->config_.out_c ,1, 1 });
			}
		}

		void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
				- Input single tensor rank = 4
				- Input single tensor has same input channels as configuration
			*/
			assert(inputs.size() == 1 && outputs.size() == 1);

			// Format BCHW - Batch, Channels, Height, Width
			assert(inputs[0]->rank() == 4);

			// Same input channels
			assert(inputs[0]->shape(1) == config_.in_c);

			config_.in_h = inputs[0]->shape(2);

			config_.in_w = inputs[0]->shape(3);

			// Height 
			config_.out_h = size_t((config_.in_h + 2 * config_.p_h - config_.k_h - (config_.k_h - 1) * (config_.d_h - 1)) / config_.s_h) + 1;

			// Width
			config_.out_w = size_t((config_.in_w + 2 * config_.p_w - config_.k_w - (config_.k_w - 1) * (config_.d_w - 1)) / config_.s_w) + 1;

			// y - Output tensor of shapes[batch, out_c, out_h, out_w]
			outputs[0]->reshape({ inputs[0]->shape(0), config_.out_c, config_.out_h, config_.out_w });
		}

		void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			// Colum buffer. Reshape like {i_c * k_h * k_w, o_h * o_w}
			col_.reshape({ inputs[0]->shape(1) * config_.k_h * config_.k_w,  outputs[0]->shape(2) * outputs[0]->shape(3) });

			T * x = inputs[0]->data();
			T * y = outputs[0]->data();

			T * w = this->parameters_[0]->data(); //W [out_c, in_c, k_h, k_w]
			T * b = this->parameters_[1]->data(); //b [out_c]

			T * cols = col_.data();

			conv2d_layer_forward<T>(
				outputs[0]->shape(0),
				config_.in_c, config_.in_h, config_.in_w,
				config_.out_c, config_.out_h, config_.out_w,
				config_.k_h, config_.k_w,
				config_.s_h, config_.s_w,
				config_.p_h, config_.p_w,
				config_.d_h, config_.d_w,
				x, w, b, cols, y
				);

		}

		void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			// cols shape in_c * k_h * k_w * out_h * out_w
			col_.reshape({ inputs[0]->shape(1) * config_.k_h * config_.k_w,  outputs[0]->shape(2) * outputs[0]->shape(3) });

			T * x = inputs[0]->data();

			T * w = this->parameters_[0]->data(); //W [out_c, in_c, k_h, k_w]
			T * b = this->parameters_[1]->data(); //b [out_c]

			T * cols = col_.data();

			// Grads
			T * dw = this->parameters_[0]->grad(); // [out_c, in_c, k_h, k_w]
			T * db = this->parameters_[1]->grad(); // [out_c]
			T * dx = inputs[0]->grad();  // [batch, in_c, in_h, in_w]
			T * dy = outputs[0]->grad(); // [batch, out_c, out_h, out_w]

			conv2d_layer_backward<T>(
				outputs[0]->shape(0),
				inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3),
				outputs[0]->shape(1), outputs[0]->shape(2), outputs[0]->shape(3),
				config_.k_h, config_.k_w,
				config_.s_h, config_.s_w,
				config_.p_h, config_.p_w,
				config_.d_h, config_.d_w,
				x, w, b, cols,
				dy, dw, db, dx
				);

		}

	private:

		Tensor<T> col_;

	};

	template<typename T>
	class Conv2DTransposedLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "conv_2d_transposed_layer"; };

		explicit Conv2DTransposedLayer(
			int in_channels, int out_channels,
			IntList<2> kernel,
			IntList<2> stride = 1,
			IntList<2> padding = 0,
			IntList<2> dilation = 1,
			bool has_bias = true
		)
		{
			this->config_.in_c = in_channels;
			this->config_.out_c = out_channels;

			this->config_.k_h = kernel[0];
			this->config_.k_w = kernel[1];
			this->config_.s_h = stride[0];
			this->config_.s_w = stride[1];
			this->config_.p_h = padding[0];
			this->config_.p_w = padding[1];
			this->config_.d_h = dilation[0];
			this->config_.d_w = dilation[1];

			this->config_.has_bias = has_bias;

			build();
		}

		Conv2DTransposedLayer(
			int in_channels, int out_channels,
			int kernel_height, int kernel_width,
			int stride_height = 1, int stride_width = 1,
			int padding_height = 0, int padding_width = 0,
			int dilation_height = 1, int dilation_width = 1,
			bool has_bias = true)
		{
			this->config_.in_c = in_channels;
			this->config_.out_c = out_channels;

			this->config_.k_h = kernel_height;
			this->config_.k_w = kernel_width;
			this->config_.s_h = stride_height;
			this->config_.s_w = stride_width;
			this->config_.p_h = padding_height;
			this->config_.p_w = padding_width;
			this->config_.d_h = dilation_height;
			this->config_.d_w = dilation_width;

			this->config_.has_bias = has_bias;

			build();
		}

		// Layer Setup
		void build()
		{
			// Define number of parameters that will be stores (weights and/or bias)
			int n_paramters = this->config_.has_bias ? 2 : 1;
			this->parameters_.resize(n_paramters);

			// weight - 4D Tensor of learnable weights of shapes[out_c, in_c, k_h, k_w]
			this->parameters_[0] = new Tensor<T>({
				this->config_.out_c,  // Output channels
				this->config_.in_c,   // Input channels 
				this->config_.k_h,    // Kernel height
				this->config_.k_w     // Kernel width
				});

			// Initialize Weights
			T k = T(1) / (this->config_.in_c * this->config_.k_h * this->config_.k_w);

			auto w = this->parameters_[0]->data();

			for (size_t i = 0; i < this->parameters_[0]->size(); i++, w++)
			{
				*w = get_random_uniform<T>(-sqrt(k), sqrt(k));
			}

			// bias - 1D Tensor of bias of shapes[out_c]
			if (this->config_.has_bias)
			{
				this->parameters_[1] = new Tensor<T>({ this->config_.out_c ,1, 1 });
			}
		}

		void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
				- Input single tensor rank = 4
				- Input single tensor has same input channels as configuration
			*/
			assert(inputs.size() == 1 && outputs.size() == 1);

			// Format BCHW - Batch, Channels, Height, Width
			assert(inputs[0]->rank() == 4);

			// Same input channels
			assert(inputs[0]->shape(1) == config_.in_c);

			config_.in_h = inputs[0]->shape(2);

			config_.in_w = inputs[0]->shape(3);

			// Height 
			config_.out_h = size_t((config_.in_h - 1) * config_.s_h - 2 * config_.p_h + config_.d_h * (config_.k_h - 1) + 1);

			// Width
			config_.out_w = size_t((config_.in_w - 1) * config_.s_w - 2 * config_.p_w + config_.d_w * (config_.k_w - 1) + 1);

			// y - Output tensor of shapes[batch, out_c, out_h, out_w]
			outputs[0]->reshape({ inputs[0]->shape(0), config_.out_c, config_.out_h, config_.out_w });
		}

		void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			// Colum buffer. Reshape like {i_c * k_h * k_w, o_h * o_w}
			col_.reshape({ inputs[0]->shape(1) * config_.k_h * config_.k_w,  outputs[0]->shape(2) * outputs[0]->shape(3) });

			T * x = inputs[0]->data();
			T * y = outputs[0]->data();

			T * w = this->parameters_[0]->data(); //W [out_c, in_c, k_h, k_w]
			T * b = this->parameters_[1]->data(); //b [out_c]

			T * cols = col_.data();

			conv2d_transposed_layer_forward<T>(
				outputs[0]->shape(0),
				config_.in_c, config_.in_h, config_.in_w,
				config_.out_c, config_.out_h, config_.out_w,
				config_.k_h, config_.k_w,
				config_.s_h, config_.s_w,
				config_.p_h, config_.p_w,
				config_.d_h, config_.d_w,
				x, w, b, cols, y
				);

		}

		void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			// cols shape in_c * k_h * k_w * out_h * out_w
			col_.reshape({ inputs[0]->shape(1) * config_.k_h * config_.k_w,  outputs[0]->shape(2) * outputs[0]->shape(3) });

			T * x = inputs[0]->data();

			T * w = this->parameters_[0]->data(); //W [out_c, in_c, k_h, k_w]
			T * b = this->parameters_[1]->data(); //b [out_c]

			T * cols = col_.data();

			// Grads
			T * dw = this->parameters_[0]->grad(); // [out_c, in_c, k_h, k_w]
			T * db = this->parameters_[1]->grad(); // [out_c]
			T * dx = inputs[0]->grad();  // [batch, in_c, in_h, in_w]
			T * dy = outputs[0]->grad(); // [batch, out_c, out_h, out_w]

			conv2d_transposed_layer_backward<T>(
				outputs[0]->shape(0),
				inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3),
				outputs[0]->shape(1), outputs[0]->shape(2), outputs[0]->shape(3),
				config_.k_h, config_.k_w,
				config_.s_h, config_.s_w,
				config_.p_h, config_.p_w,
				config_.d_h, config_.d_w,
				x, w, b, cols,
				dy, dw, db, dx
				);

		}

	private:

		Tensor<T> col_;

	};

	template<typename T>
	class MaxPooling2DLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "max_pooling_2d_layer"; };

		MaxPooling2DLayer(
			IntList<2> kernel = 2,
			IntList<2> stride = 2,
			IntList<2> padding = 0,
			IntList<2> dilation = 1
		)
		{
			this->config_.k_h = kernel[0];
			this->config_.k_w = kernel[1];
			this->config_.s_h = stride[0];
			this->config_.s_w = stride[1];
			this->config_.p_h = padding[0];
			this->config_.p_w = padding[1];
			this->config_.d_h = dilation[0];
			this->config_.d_w = dilation[1];
		}

		MaxPooling2DLayer(
			int kernel_height = 2, int kernel_width = 2,
			int stride_height = 2, int stride_width = 2,
			int padding_height = 0, int padding_width = 0,
			int dilation_height = 1, int dilation_width = 1)
		{
			this->config_.k_h = kernel_height;
			this->config_.k_w = kernel_width;
			this->config_.s_h = stride_height;
			this->config_.s_w = stride_width;
			this->config_.p_h = padding_height;
			this->config_.p_w = padding_width;
			this->config_.d_h = dilation_height;
			this->config_.d_w = dilation_width;
		}

		~MaxPooling2DLayer() {};

		virtual void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
				- Input single tensor rank = 4
			*/

			assert(inputs.size() == 1 && outputs.size() == 1);

			// Format BCHW - Batch, Channels, Height, Width
			assert(inputs[0]->rank() == 4);

			config_.in_h = inputs[0]->shape(2);

			config_.in_w = inputs[0]->shape(3);

			// Height 
			config_.out_h = size_t((config_.in_h + 2 * config_.p_h - config_.k_h - (config_.k_h - 1) * (config_.d_h - 1)) / config_.s_h) + 1;

			// Width
			config_.out_w = size_t((config_.in_w + 2 * config_.p_w - config_.k_w - (config_.k_w - 1) * (config_.d_w - 1)) / config_.s_w) + 1;

			// Reshape output tensor
			outputs[0]->reshape({ inputs[0]->shape(0), inputs[0]->shape(1), config_.out_h, config_.out_w });

			// Reshape indices of argmax
			this->indices_.reshape(outputs[0]->shapes());

			// Reset max value indices 
			memset(indices_.data(), 0, sizeof(T) * indices_.size());
		};

		virtual void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * y = outputs[0]->data();
			T * indices = indices_.data();

			maxpool2d_layer_forward<T>(
				inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3),
				outputs[0]->shape(2), outputs[0]->shape(3),
				config_.k_h, config_.k_w,
				config_.s_h, config_.s_w,
				config_.p_h, config_.p_w,
				config_.d_h, config_.d_w,
				x, indices, y
				);

		};

		virtual void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * dx = inputs[0]->grad();  // [batch, channels, in_h, in_w]
			T * dy = outputs[0]->grad(); // [batch, channels, out_h, out_w]
			T * indices = indices_.data();

			maxpool2d_layer_backward<T>(
				inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3),
				outputs[0]->shape(2), outputs[0]->shape(3),
				config_.k_h, config_.k_w,
				config_.s_h, config_.s_w,
				config_.p_h, config_.p_w,
				config_.d_h, config_.d_w,
				indices, dy, dx
				);

		};

		Tensor<T>& indices() { return indices_; };

		void set_indices(const Tensor<T>& indices) { indices_ = indices; };

	private:

		Tensor<T> indices_;

	};

	template<typename T>
	class AvgPooling2DLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "avg_pooling_2d_layer"; };

		AvgPooling2DLayer(
			IntList<2> kernel = 2,
			IntList<2> stride = 2,
			IntList<2> padding = 0,
			IntList<2> dilation = 1
		)
		{
			this->config_.k_h = kernel[0];
			this->config_.k_w = kernel[1];
			this->config_.s_h = stride[0];
			this->config_.s_w = stride[1];
			this->config_.p_h = padding[0];
			this->config_.p_w = padding[1];
			this->config_.d_h = dilation[0];
			this->config_.d_w = dilation[1];
		}

		AvgPooling2DLayer(
			int kernel_height = 2, int kernel_width = 2,
			int stride_height = 2, int stride_width = 2,
			int padding_height = 0, int padding_width = 0,
			int dilation_height = 1, int dilation_width = 1)
		{
			this->config_.k_h = kernel_height;
			this->config_.k_w = kernel_width;
			this->config_.s_h = stride_height;
			this->config_.s_w = stride_width;
			this->config_.p_h = padding_height;
			this->config_.p_w = padding_width;
			this->config_.d_h = dilation_height;
			this->config_.d_w = dilation_width;
		}

		~AvgPooling2DLayer() {};

		virtual void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
				- Input single tensor rank = 4
			*/

			assert(inputs.size() == 1 && outputs.size() == 1);

			// Format BCHW - Batch, Channels, Height, Width
			assert(inputs[0]->rank() == 4);

			config_.in_h = inputs[0]->shape(2);

			config_.in_w = inputs[0]->shape(3);

			// Height 
			config_.out_h = size_t((config_.in_h + 2 * config_.p_h - config_.k_h - (config_.k_h - 1) * (config_.d_h - 1)) / config_.s_h) + 1;

			// Width
			config_.out_w = size_t((config_.in_w + 2 * config_.p_w - config_.k_w - (config_.k_w - 1) * (config_.d_w - 1)) / config_.s_w) + 1;

			// Reshape output tensor
			outputs[0]->reshape({ inputs[0]->shape(0), inputs[0]->shape(1), config_.out_h, config_.out_w });

		};

		virtual void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * y = outputs[0]->data();

			avgpool2d_layer_forward<T>(
				inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3),
				outputs[0]->shape(2), outputs[0]->shape(3),
				config_.k_h, config_.k_w,
				config_.s_h, config_.s_w,
				config_.p_h, config_.p_w,
				config_.d_h, config_.d_w,
				x, y
				);

		};

		virtual void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * dx = inputs[0]->grad();  // [batch, channels, in_h, in_w]
			T * dy = outputs[0]->grad(); // [batch, channels, out_h, out_w]

			avgpool2d_layer_backward<T>(
				inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3),
				outputs[0]->shape(2), outputs[0]->shape(3),
				config_.k_h, config_.k_w,
				config_.s_h, config_.s_w,
				config_.p_h, config_.p_w,
				config_.d_h, config_.d_w,
				dy, dx
				);

		};

	};

	template<typename T>
	class BatchNorm2DLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "batch_norm_2d_layer"; };

		BatchNorm2DLayer(size_t n_features, T momentum = 0.99, T epsilon = 0.001, bool is_training = false)
		{
			this->config_.n_features = n_features;
			this->config_.momentum = momentum;
			this->config_.epsilon = epsilon;
			this->config_.is_training = is_training;

			build();
		}

		~BatchNorm2DLayer() {};

		// Layer Setup
		void build()
		{
			// Initialize Gamma and Beta parameters
			this->parameters_.resize(2);

			parameters_[0] = new Tensor<T>({ this->config_.n_features }); // Gamma
			parameters_[1] = new Tensor<T>({ this->config_.n_features }); // Beta

			mean_.reshape({ this->config_.n_features });
			variance_.reshape({ this->config_.n_features });

			moving_mean_.reshape({ this->config_.n_features });
			moving_variance_.reshape({ this->config_.n_features });

			// Initialize Gamma with ones
			for (size_t i = 0; i < parameters_[0]->size(); i++)
			{
				(*parameters_[0])[i] = T(1);
			}

			// Initialize Moving Variance with ones
			for (size_t i = 0; i < moving_variance_.size(); i++)
			{
				moving_variance_[i] = T(1);
			}

		}

		virtual void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
				- Rank = 4
			*/

			assert(inputs.size() == 1 && outputs.size() == 1);

			// Format BCHW - Batch, Channels, Height, Width
			assert(inputs[0]->rank() == 4);

			outputs[0]->reshape(inputs[0]->shapes());

			norm_.reshape(inputs[0]->shapes());

		}

		virtual void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			size_t shape = inputs[0]->shape(1);
			size_t stride = inputs[0]->stride(1);

			T * x = inputs[0]->data();
			T * y = outputs[0]->data();

			T * mu = mean_.data();
			T * var = variance_.data();
			T * norm = norm_.data();

			T * gamma = this->parameters_[0]->data();
			T * beta = this->parameters_[1]->data();

			T * moving_mean = moving_mean_.data();
			T * moving_variance = moving_variance_.data();

			batch_norm_forward<T>(
				inputs[0]->size(),
				shape, stride,
				this->config_.momentum, this->config_.epsilon,
				x,
				mu, var, norm, gamma, beta, moving_mean, moving_variance,
				y,
				this->config_.is_training);

		}

		virtual void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			size_t shape = inputs[0]->shape(1);
			size_t stride = inputs[0]->stride(1);

			T * x = inputs[0]->data();
			T * dx = inputs[0]->grad();
			T * dy = outputs[0]->grad();

			T * mu = mean_.data();
			T * dmu = mean_.grad();
			T * var = variance_.data();
			T * dvar = variance_.grad();
			T * norm = norm_.data();

			T * gamma = this->parameters_[0]->data();
			T * dgamma = this->parameters_[0]->grad();
			T * beta = this->parameters_[1]->data();
			T * dbeta = this->parameters_[1]->grad();

			batch_norm_backward<T>(
				inputs[0]->size(),
				shape, stride, this->config_.epsilon,
				x,
				mu, var, norm, gamma, beta, dmu, dvar, dgamma, dbeta,
				dy, dx);
		}

	private:

		Tensor<T> mean_;
		Tensor<T> variance_;

		Tensor<T> moving_mean_;
		Tensor<T> moving_variance_;

		Tensor<T> norm_;
	};

	template<typename T>
	class DroputLayer : public ILayer<T>
	{
	public:
		DroputLayer(T alpha, bool is_training = false)
		{
			this->config_.alpha = alpha;
			this->config_.is_training = is_training;
		};

		~DroputLayer() {};

		// Reshape
		virtual void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
			*/
			assert(inputs.size() == 1 && outputs.size() == 1);

			outputs[0]->reshape(inputs[0]->shapes());

			mask_.reshape(inputs[0]->shapes());
		};

		// Feed Forward
		virtual void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * y = outputs[0]->data();
			T * mask = mask_.data();

			dropout_layer_forward<T>(inputs[0]->size(), this->config_.alpha, x, y, mask, this->config_.is_traning);
		};

		// Backward
		virtual void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * dx = inputs[0]->grad();
			T * dy = outputs[0]->grad();
			T * mask = mask_.data();

			dropout_layer_backward<T>(inputs[0]->size(), this->config_.alpha, dy, mask, dx);
		};


	private:
		Tensor<T> mask_;

	};

	/**
	 * Linear Layer
	*/
	template<typename T>
	class LinearLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "linear_layer"; };

		void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
			*/

			assert(inputs.size() == 1 && outputs.size() == 1);

			outputs[0]->reshape(inputs[0]->shapes());

		}

		void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * y = outputs[0]->data();

			memcpy(y, x, inputs[0]->size() * sizeof(T));
		}

		void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * dx = inputs[0]->grad();
			T * dy = outputs[0]->grad();

			memcpy(dx, dy, inputs[0]->size() * sizeof(T));
		}

	};

	/**
	 * Sigmoid Layer
	*/
	template<typename T>
	class SigmoidLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "sigmoid_layer"; };

		void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
			*/

			assert(inputs.size() == 1 && outputs.size() == 1);

			outputs[0]->reshape(inputs[0]->shapes());

		}

		void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * y = outputs[0]->data();

			sigmoid_forward(inputs[0]->size(), x, y);
		}

		void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * dx = inputs[0]->grad();
			T * dy = outputs[0]->grad();

			sigmoid_backward(inputs[0]->size(), x, dy, dx);
		}

	};

	/**
	 * Tanh Layer
	*/
	template<typename T>
	class TanhLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "tanh_layer"; };

		void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
			*/

			assert(inputs.size() == 1 && outputs.size() == 1);

			outputs[0]->reshape(inputs[0]->shapes());

		}

		void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * y = outputs[0]->data();

			tanh_forward(inputs[0]->size(), x, y);
		}

		void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * dx = inputs[0]->grad();
			T * dy = outputs[0]->grad();

			tanh_backward(inputs[0]->size(), x, dy, dx);
		}

	};


	/**
	 * ReLU Layer
	*/
	template<typename T>
	class ReLULayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "relu_layer"; };

		void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
			*/

			assert(inputs.size() == 1 && outputs.size() == 1);

			outputs[0]->reshape(inputs[0]->shapes());

		}

		void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * y = outputs[0]->data();

			relu_forward(inputs[0]->size(), x, y);
		}

		void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * dx = inputs[0]->grad();
			T * dy = outputs[0]->grad();

			relu_backward(inputs[0]->size(), x, dy, dx);
		}

	};

	/**
	 * Softmax Layer
	*/
	template<typename T>
	class SoftmaxLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "softmax_layer"; };

		SoftmaxLayer(int axis = -1)
		{
			this->config_.axis = axis;
		}

		void reshape(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size = 1
				- Output tensor vector size = 1
				- Input single tensor rank > axis
			*/

			assert(inputs.size() == 1 && outputs.size() == 1);

			assert(static_cast<int>(inputs[0]->rank()) > this->config_.axis);

			outputs[0]->reshape(inputs[0]->shapes());
		}

		void forward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * x = inputs[0]->data();
			T * y = outputs[0]->data();

			size_t shape = inputs[0]->shape(this->config_.axis);
			size_t stride = inputs[0]->stride(this->config_.axis);

			softmax_forward(x, inputs[0]->size(), shape, stride, y);
		}

		void backward_impl(const std::vector<Tensor<T>*>& inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			T * y = outputs[0]->data();
			T * dx = inputs[0]->grad();
			T * dy = outputs[0]->grad();

			size_t shape = inputs[0]->shape(this->config_.axis);
			size_t stride = inputs[0]->stride(this->config_.axis);

			softmax_backward(y, inputs[0]->size(), shape, stride, dy, dx);
		}
	};

	/**
	 * Concatenate Layer
	*/
	template<typename T>
	class ConcatLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "concat_layer"; };

		ConcatLayer(int axis = 0) { this->config_.axis = axis; };
		~ConcatLayer() {};

		// Reshape
		void reshape(const std::vector<Tensor<T>*>&  inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- All input tensors must have same rank
				- All dimensions on input tensors have to be equal excet on concatenation axis
			*/

			// Assure tensors have same valid parameters
			for (size_t i = 1; i < inputs.size(); i++)
			{
				// Same shape
				assert(inputs[0]->rank() == inputs[i]->rank());

				// Same dimension except on axis concatenation 
				for (size_t j = 0; j < inputs[0]->rank(); j++)
				{
					if (j == this->config_.axis)
						continue;

					assert(inputs[0]->shape(j) == inputs[i]->shape(j));
				}
			}

			// Reshape single concatenate output tensor
			size_t output_size_ = inputs[0]->size();
			size_t output_axis_size = 0;

			output_shapes_ = inputs[0]->shapes();
			output_shapes_[this->config_.axis] = 0;

			for (size_t i = 0; i < inputs.size(); i++)
			{
				output_size_ += inputs[i]->size();
				output_shapes_[this->config_.axis] += inputs[i]->shape(this->config_.axis);
			}

			outputs[0]->reshape(output_shapes_);
		};

		// Feed Forward
		void forward_impl(const std::vector<Tensor<T>*>&  inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			int offset = 0;
			for (size_t i = 0; i < inputs.size(); i++)
			{
				memcpy(outputs[0]->data() + offset, inputs[i]->data(), inputs[i]->size() * sizeof(T));
				offset += inputs[i]->size();
			}

		};

		// Feed Forward
		void backward_impl(const std::vector<Tensor<T>*>&  inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			assert(outputs[0]->shapes() == output_shapes_);

			int offset = 0;
			for (size_t i = 0; i < inputs.size(); i++)
			{
				memcpy(inputs[i]->grad(), outputs[0]->grad() + offset, inputs[i]->size() * sizeof(T));
				offset += inputs[i]->size();
			}
		};

	private:

		std::vector<size_t> output_shapes_;

	};

	/**
	 * Add Layer
	*/
	template<typename T>
	class AddLayer : public ILayer<T>
	{
	public:

		virtual const std::string type() const override { return "add_layer"; };

		AddLayer() {};
		~AddLayer() {};

		// Reshape
		virtual void reshape(const std::vector<Tensor<T>*>&  inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			/*
			Layer Restrictions:
				- Input tensor vector size > 1
				- All dimensions on input tensors have to be equal
				- Output tensor vector size == 1
			*/

			assert(inputs.size() > 1);

			for (size_t i = 1; i < inputs.size(); i++)
			{
				assert(inputs[i]->shapes() == inputs[0]->shapes());
			}

			assert(outputs.size() == 1);

			outputs[0]->reshape(inputs[0]->shapes());
		};

		// Feed Forward
		virtual void forward_impl(const std::vector<Tensor<T>*>&  inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			for (size_t i = 0; i < inputs.size(); i++)
			{
				axpby<T>(inputs[i]->size(), T(1), inputs[i]->data(), 1, (i > 0), outputs[0]->data(), 1);
			}
		};

		// Backward
		virtual void backward_impl(const std::vector<Tensor<T>*>&  inputs, const std::vector<Tensor<T>*>& outputs) override
		{
			for (size_t i = 0; i < inputs.size(); i++)
			{
				inputs[i]->set_grad(outputs[0]->size(), outputs[0]->grad());
			}
		};

	};

	/********************************************************************
	 Loss
	*********************************************************************/
	template<typename T>
	class ILoss
	{
	public:

		virtual T loss(Tensor<T> * inputs, Tensor<T> * targets) = 0;

		virtual void cost(Tensor<T> * inputs, Tensor<T> * targets) = 0;

	};

	template<typename T>
	class MSELoss : public ILoss<T>
	{
	public:

		T loss(Tensor<T> * inputs, Tensor<T> * targets) override
		{
			T error = mse_loss_layer_forward(inputs->size(), inputs->data(), targets->data());

			return error;
		}

		void cost(Tensor<T> * inputs, Tensor<T> * targets) override
		{
			mse_loss_layer_backward(inputs->size(), inputs->data(), targets->data(), inputs->grad());
		}

	};

	template<typename T>
	class HuberLoss : public ILoss<T>
	{
	public:

		HuberLoss(T delta = T(1)) :
			delta_(delta)
		{

		}

		T loss(Tensor<T> * inputs, Tensor<T> * targets) override
		{
			T error = huber_loss_layer_forward(inputs->size(), delta_, inputs->data(), targets->data());

			return error;
		}

		void cost(Tensor<T> * inputs, Tensor<T> * targets) override
		{
			huber_loss_layer_backward(inputs->size(), delta_, inputs->data(), targets->data(), inputs->grad());
		}

	private:
		T delta_;

	};

	template<typename T>
	class BinaryCrossEntropyLoss : public ILoss<T>
	{
	public:

		T loss(Tensor<T> * inputs, Tensor<T> * targets) override
		{
			T error = binary_cross_entropy_loss_layer_forward(inputs->size(), inputs->data(), targets->data());

			return error;
		}

		void cost(Tensor<T> * inputs, Tensor<T> * targets) override
		{
			binary_cross_entropy_loss_layer_backward(inputs->size(), inputs->data(), targets->data(), inputs->grad());
		}

	};

	template<typename T>
	class CrossEntropyLoss : public ILoss<T>
	{
	public:

		CrossEntropyLoss(int axis = -1)
		{
			axis_ = axis;
		}

		T loss(Tensor<T> * inputs, Tensor<T> * targets) override
		{
			size_t shape = inputs->shape(axis_);
			size_t stride = inputs->stride(axis_);

			T error = cross_entropy_loss_layer_forward(inputs->size(), shape, stride, inputs->data(), targets->data());

			return error;
		}

		void cost(Tensor<T> * inputs, Tensor<T> * targets) override
		{
			size_t shape = inputs->shape(axis_);
			size_t stride = inputs->stride(axis_);

			cross_entropy_loss_layer_backward(inputs->size(), shape, stride, inputs->data(), targets->data(), inputs->grad());
		}

	private:
		int axis_;

	};

	template<typename T>
	class CrossEntropyFromLogitsLoss : public ILoss<T>
	{
	public:

		CrossEntropyFromLogitsLoss(int axis = -1) :
			axis_(axis)
		{

		}

		T loss(Tensor<T> * inputs, Tensor<T> * targets) override
		{
			size_t shape = inputs->shape(axis_);
			size_t stride = inputs->stride(axis_);

			T error = cross_entropy_loss_from_logits_layer_forward(inputs->size(), shape, stride, inputs->data(), targets->data());

			return error;
		}

		void cost(Tensor<T> * inputs, Tensor<T> * targets) override
		{
			size_t shape = inputs->shape(axis_);
			size_t stride = inputs->stride(axis_);

			cross_entropy_loss_from_logits_layer_backward(inputs->size(), shape, stride, inputs->data(), targets->data(), inputs->grad());
		}

	private:
		int axis_;

	};

	/********************************************************************
	 Optimizers
	*********************************************************************/
	template<typename T>
	class IOptimizer
	{
	public:
		IOptimizer() {};


		virtual void step(const std::vector<Tensor<T> *>& parameters) = 0;

	};

	/**
	 * Stochastic Gradient Descent Optimizer
	 * w = w - learning rate * g
	*/
	template<typename T>
	class SGDOptimizer : public IOptimizer<T>
	{
	public:
		explicit SGDOptimizer(T lr = 0.001, T mom = 0.0) : lr_(lr), mom_(mom)
		{
			v_.resize(parameters.size());
		};

		void step(const std::vector<Tensor<T>*>& parameters)
		{
			size_t i;

			for (i = 0; i < this->parameters_.size(); i++)
			{

				if (v_[i].size() != this->parameters_[i]->size())
				{
					v_[i].resize(this->parameters_[i]->size());
				}

				T * w = this->parameters_[i]->data();
				T * g = this->parameters_[i]->grad();
				T * v = v_[i].data();

				sgd_optimizer<T>(this->parameters_[i]->size(), lr_, mom_, g, v, w);

			}
		}

	private:
		T lr_;
		T mom_;

		std::vector<std::vector<T>> v_;

	};


	template<typename T>
	class RMSpropOptimizer : public IOptimizer<T>
	{
	public:
		explicit RMSpropOptimizer(T lr = 0.001, T rho = 0.9, T mom = 0.0, T eps = 1e-07) :
			lr_(lr),
			rho_(rho),
			mom_(mom),
			eps_(eps)
		{
			v_.resize(parameters.size());
		};

		void step(const std::vector<Tensor<T>*>& parameters)
		{
			size_t i;

			for (i = 0; i < this->parameters_.size(); i++)
			{

				if (v_[i].size() != this->parameters_[i]->size())
				{
					v_[i].resize(this->parameters_[i]->size());
				}

				T * w = this->parameters_[i]->data();
				T * g = this->parameters_[i]->grad();
				T * v = v_[i].data();

				rms_prop_optimizer<T>(this->parameters_[i]->size(), lr_, rho_, mom_, eps_, g, v, w);

			}
		}

	private:
		T lr_;
		T rho_;
		T mom_;
		T eps_;

		std::vector<std::vector<T>> v_;

	};

	template<typename T>
	class AdamOptimizer : public IOptimizer<T>
	{
	public:
		explicit AdamOptimizer(T lr = 0.001, T beta_1 = 0.9, T beta_2 = 0.999, T eps = 1e-07) :
			lr_(lr),
			beta_1_(beta_1),
			beta_2_(beta_2),
			eps_(eps)
		{
			v_.resize(parameters.size());
			m_.resize(parameters.size());
		};

		void step(const std::vector<Tensor<T>*>& parameters)
		{
			size_t i;

			for (i = 0; i < this->parameters_.size(); i++)
			{

				if (v_[i].size() != this->parameters_[i]->size())
				{
					v_[i].resize(this->parameters_[i]->size());
				}

				if (m_[i].size() != this->parameters_[i]->size())
				{
					m_[i].resize(this->parameters_[i]->size());
				}

				T * w = this->parameters_[i]->data();
				T * g = this->parameters_[i]->grad();
				T * v = v_[i].data();
				T * m = m_[i].data();

				adam_optimizer<T>(this->parameters_[i]->size(), iter_, lr_, beta_1_, beta_2_, eps_, g, v, m, w);

			}

			iter_ += 1;
		}

	private:
		T lr_;
		T beta_1_;
		T beta_2_;
		T eps_;

		std::vector<std::vector<T>> v_;
		std::vector<std::vector<T>> m_;

		size_t iter_{ 1 };

	};

	/********************************************************************
	 Net
	*********************************************************************/
	template<typename T>
	class Net
	{
	public:

		Net()
		{

		}

		// Getters
		const ILayer<T> * const get_layer(const std::string& name) const
		{
			return layers_[layer_names_to_indices_[name]];
		}

		const ILayer<T> * const get_layer(size_t index) const
		{
			return layers_[index];
		}

		const Tensor<T> * const get_tensor(const std::string& name) const
		{
			return tensors_[tensor_names_to_indices_[name]];
		}

		const Tensor<T> * const get_tensor(size_t index) const
		{
			return tensors_[index];
		}

		const std::vector<Tensor<T>*> get_tensors() const
		{
			return tensors_;
		}

		const std::vector<Tensor<T>*> get_layer_inputs(const std::string& name) const
		{
			return inputs_[layer_names_to_indices_[name]];
		};

		const std::vector<Tensor<T>*> get_layer_inputs(size_t index) const
		{
			return inputs_[size_t];
		};

		const std::vector<Tensor<T>*> get_layer_outputs(const std::string& name) const
		{
			return outputs_[layer_names_to_indices_[name]];
		};

		const std::vector<Tensor<T>*> get_layer_outputs(size_t index) const
		{
			return outputs_[size_t];
		};

		const std::vector<Tensor<T>*> get_net_inputs() const { return net_inputs_; }
		const std::vector<Tensor<T>*> get_net_outputs() const { return net_outputs_; }

		std::vector<Tensor<T>*>& get_parameters() { return parameters_; }

		// Setters
		void set_net_inputs(const std::vector<Tensor<T>>& inputs)
		{
			// Assert number of tensors for net and inputs are equals
			assert(inputs.size() == net_inputs_.size());

			//Assert the tensors are in Batch x Dim 1 x Dim 2...Dim N format
			for (size_t i = 0; i < inputs.size(); i++)
			{
				assert(inputs[i].shapes().size() >= 2);

				// Make a copy so we don't overrite original tensors. Don't change pointer!
				*net_inputs_[i] = inputs[i];
			}
		}

		void set_parameters(const std::vector<Tensor<T>*>& parameters)
		{
			// Assert number of tensors for net and inputs are equals
			assert(parameters_.size() == parameters.size());

			//Assert the tensors are in Batch x Dim 1 x Dim 2...Dim N format
			for (size_t i = 0; i < parameters.size(); i++)
			{
				// Make a copy so we don't overrite original tensors. Don't change pointer!
				*parameters_[i] = *parameters[i];
			}
		}

		size_t add_layer(ILayer<T> * layer, const std::string& layer_name)
		{
			size_t layer_index = add_layer(layer, layer_name, {});

			return layer_index;
		}

		size_t add_layer(
			ILayer<T> * layer,
			const std::string& layer_name,
			const std::vector<std::string>& prev_layers)
		{

			std::vector<std::pair<std::string, size_t>> prev_layers_with_tensor_indices;

			for (size_t i = 0; i < prev_layers.size(); i++)
			{
				prev_layers_with_tensor_indices.push_back({ prev_layers[i], 0 });
			}

			size_t layer_index = add_layer_with_indices(layer, layer_name, prev_layers_with_tensor_indices);

			return layer_index;
		}

		size_t add_layer_with_indices(
			ILayer<T> * layer,
			const std::string& layer_name,
			const std::vector<std::pair<std::string, size_t>>& prev_layers_with_tensor_indices)
		{
			size_t i, j;

			// Create layer index
			size_t layer_index = layers_.size();

			// Update mapping between layer name and layer index
			layer_names_.push_back(layer_name);
			layer_names_to_indices_[layer_name] = layer_index;

			// Insert layer on layers_ where the layer index is the same as the layer position inside the vector.
			layers_.push_back(layer);

			// Initialize input and output tensor for layer
			std::vector<Tensor<T>*> input_tensors;
			std::vector<Tensor<T>*> output_tensors;

			inputs_.push_back(input_tensors);
			outputs_.push_back(output_tensors);

			// Update tensor graph
			std::vector<size_t> prev_list;
			layer_graph_prev_.push_back(prev_list);

			std::vector<size_t> next_list;
			layer_graph_next_.push_back(next_list);

			// Create layer input tensors and update tensor graph
			if (prev_layers_with_tensor_indices.empty())
			{
				// Create input tensor
				Tensor<T> * tensor = new Tensor<T>();

				// Update Layer input tensors
				inputs_[layer_index].push_back(tensor);

				// Update Net input tensors with same pointer since there is no precendent layer
				net_inputs_.push_back(tensor);

				// Create tensor name
				std::string tensor_name = "tensor_" + std::to_string(layer_index) + "_" + layer_name;

				tensor_names_to_indices_[tensor_name] = tensors_.size();

				// Update Net tensors
				tensors_.push_back(tensor);

			}
			else
			{
				for (i = 0; i < prev_layers_with_tensor_indices.size(); i++)
				{
					size_t prev_layer_index = layer_names_to_indices_[prev_layers_with_tensor_indices[i].first];
					size_t tensor_index = prev_layers_with_tensor_indices[i].second;

					// Update Layer input tensors
					inputs_[layer_index].push_back(outputs_[prev_layer_index][tensor_index]);

					// Update layer graph connections
					layer_graph_prev_[layer_index].push_back(prev_layer_index);
					layer_graph_next_[prev_layer_index].push_back(layer_index);
				}
			}

			// Create layer output tensors
			for (i = 0; i < layer->get_output_count(); i++)
			{
				Tensor<T> * tensor = new Tensor<T>();

				outputs_[layer_index].push_back(tensor);

				std::string tensor_name = "tensor_" + std::to_string(layer_index) + "_" + layer_name + "_" + std::to_string(tensors_.size());

				tensor_names_to_indices_[tensor_name] = tensors_.size();

				// Update Net tensors
				tensors_.push_back(tensor);

			}

			// Update net output tensors

			net_outputs_.clear();

			start_indices_.clear();
			for (i = 0; i < layer_graph_prev_.size(); i++)
			{
				if (layer_graph_prev_[i].size() == 0)
				{
					start_indices_.push_back(i);
				}
			}

			end_indices_.clear();

			for (i = 0; i < layer_graph_next_.size(); i++)
			{
				if (layer_graph_next_[i].size() == 0)
				{
					for (j = 0; j < outputs_[i].size(); j++)
					{
						net_outputs_.push_back(outputs_[i][j]);
					}

					end_indices_.push_back(i);
				}
			}

			for (i = 0; i < layer->parameters().size(); i++)
			{
				parameters_.push_back(layer->parameters()[i]);
			}

			return layer_index;
		}


		void forward()
		{
			for (size_t i = 0; i < layers_.size(); i++)
			{
				layers_[i]->forward(inputs_[i], outputs_[i]);
			}

		}

		std::vector<T> calculate_loss(
			const std::vector<Tensor<T>>& targets,
			const std::vector<ILoss<T>*>& losses)
		{
			std::vector<T> loss;

			std::vector<Tensor<T>*> net_targets_;

			for (size_t i = 0; i < targets.size(); i++)
			{
				net_targets_.emplace_back(const_cast<Tensor<T>*>(&targets[i]));
			}

			for (size_t i = 0; i < net_outputs_.size(); i++)
			{
				loss.push_back(losses[i]->loss(net_outputs_[i], net_targets_[i]));

				losses[i]->cost(net_outputs_[i], net_targets_[i]);
			}

			return loss;
		}

		void backward()
		{
			for (size_t i = layers_.size(); i-- > 0;)
			{
				layers_[i]->backward(inputs_[i], outputs_[i]);
			}
		}

		std::vector<Tensor<T>> predict(const std::vector<Tensor<T>>& inputs)
		{
			std::vector<Tensor<T>> outputs;

			set_net_inputs(inputs);

			forward();

			for (size_t i = 0; i < net_outputs_.size(); i++)
			{
				Tensor<T> output(net_outputs_[i]->shapes());

				output.set_data(net_outputs_[i]->size(), net_outputs_[i]->data());

				outputs.push_back(output);
			}

			return outputs;
		}

		std::vector<T> train_on_batch(
			const std::vector<Tensor<T>>& inputs,
			const std::vector<Tensor<T>>& targets,
			const std::vector<ILoss<T>*>& losses,
			IOptimizer<T> * optimizer)
		{
			// Reset Gradients
			zero_grad();

			// Set net inputs
			set_net_inputs(inputs);

			// Feed forward pass
			forward();

			// Calculate net loss and gradients with respect of target
			std::vector<T> loss = calculate_loss(targets, losses);

			// Backpropagation pass
			backward();

			// Parameters update
			optimizer->step();

			return loss;
		}

		void zero_grad()
		{
			size_t i;

			// Reset parameters gradients
			for (i = 0; i < parameters_.size(); i++)
			{
				memset(parameters_[i]->grad(), 0, sizeof(T) * parameters_[i]->size());
			}

			// Reset tensors gradients
			for (i = 0; i < tensors_.size(); i++)
			{
				memset(tensors_[i]->grad(), 0, sizeof(T) * tensors_[i]->size());
			}
		}

		void summary()
		{

			std::cout << std::left << "Layer name (type)" << std::setw(30) <<
				std::left << "Total patameters" << std::setw(15) << "\n";

			for (size_t i = 0; i < layers_.size(); i++)
			{
				std::cout << "--------------------------------------------------------------------------------------------" << "\n";
				std::cout << layer_names_[i] << " (" << layers_[i]->type() << ") ";
				std::cout << layers_[i]->get_total_parameters();

				for (size_t j = 0; j < layer_graph_prev_[i].size(); j++)
				{
					std::cout << layer_names_[layer_graph_prev_[i][j]];
				}

				std::cout << "\n";
			}
		}

	private:

		// Net parameters
		std::vector<Tensor<T>*> parameters_;

		// Net layers
		std::vector<ILayer<T>*> layers_;

		// Net tensors
		std::vector<Tensor<T>*> tensors_;

		// Layers names
		std::vector<std::string> layer_names_;

		// Layers mapping name to index
		std::map<std::string, size_t> layer_names_to_indices_;

		// Tensors mapping name to index
		std::map<std::string, size_t> tensor_names_to_indices_;

		std::vector<Tensor<T>*> net_inputs_;
		std::vector<Tensor<T>*> net_outputs_;

		// Vector of inputs and outputs for each layer
		std::vector<std::vector<Tensor<T>*>> inputs_;
		std::vector<std::vector<Tensor<T>*>> outputs_;

		// Layer adj list as graph
		std::vector<std::vector<size_t>> layer_graph_prev_;
		std::vector<std::vector<size_t>> layer_graph_next_;

		// Net head and tail
		std::vector<size_t> start_indices_;
		std::vector<size_t> end_indices_;

	};

}