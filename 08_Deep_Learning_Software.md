# Chapter 08. Deep Learning Software

## 🌱 Review

- regularization - Dropout: forward pass에서 임의의 부분 0, test time에서  marginalize out
- regularization - basic idea: train time + noise, test time + marginalize out
- transfer learning

## 🌱 CPU vs GPU

- **CPU**: central processing unit
	- small, samll size

- **GPU**: graphics card, graphics processing unit
	- rendering for computer graphics
	- big, big size
	- many power
	- deep learning: *NVIDIA*
		- cuBLAS, *cuDNN* library: don't have to code CUDA 
		- OpenCL(+ AMD, CPU)( < CUDA)

- CPU vs GPU

| | CPU | GPU |
|---|---|---|
| core | 4/6/10 | thousand (clock speed) |
| thread | 8~20 (+hyperthreading) | |

- CPU의 memory는 독립적, GPU는 병렬적
	- 병렬화 문제에서 GPU 처리량 압도적
	- GPU는 같은 문제에 대하여 병렬적으로 처리해야 함
	- Convolution 연산을 각 core에 분배시켜 연산을 빠르게 함

- GPU problem
	- Model, Weight: GPU RAM / Train Data: SSD
		- **bottleneck**
	- solution
		- data set -> RAM (if data set is small)
		- or HHD -> SSD
		- **pre-fetching**
		- buffer -> GPU data



## 🌱 Deep Learning Frameworks

I. not have to make complex Graph On Our Own

II. computating Gradient - automatically (loss and weight gradient of loss) - only have to compute forward pass, back propagation computed automatically

III. GPU efficiently - CPU -> GPU: easy

- 1. academia
	- Berkely: Caffe
	- NYU + facebook: Torch

- 2. industry
	- Facebook: Caffe2, PyTorch
	- Google: TensorFlow
	- Baidu: Paddle
	- Microsoft: CNTK
	- Amazon: MXNet

- academia to industry


