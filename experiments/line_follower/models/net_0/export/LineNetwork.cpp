#include <LineNetwork.h>
#include <Linear.h>
#include <Conv1d.h>
#include <Conv2d.h>
#include <ReLU.h>
#include <GlobalAveragePooling.h>


const int8_t LineNetwork_layer_0_weights[] = {
20, -5, 3, 13, -5, -22, 11, -53, -65, 
-64, -2, -40, -90, -24, 29, -64, -54, 51, 
92, 17, 45, 127, 69, 65, 51, 127, 114, 
-57, 32, 13, -54, -7, 59, 49, -8, 113, 
};

const int8_t LineNetwork_layer_0_bias[] = {
2, -9, -127, 49, };


const int8_t LineNetwork_layer_2_weights[] = {
127, 70, 95, 53, 
20, -13, -93, -74, 
-65, 8, -67, 13, 
5, 47, 59, 31, 
30, 19, 0, 14, 
67, 1, 7, 40, 
65, 102, -27, -3, 
127, 27, 59, 83, 
14, 107, 71, -28, 
21, 127, 87, -5, 
86, 127, 36, -127, 
101, 109, -21, -29, 
53, 91, 87, -6, 
63, 108, 67, -28, 
95, -34, -58, 3, 
18, 114, -12, 47, 
127, 55, 122, 54, 
103, 38, -7, 31, 
127, 126, 21, 126, 
127, 127, 81, 12, 
127, 119, 96, 51, 
70, 56, 92, 112, 
57, 127, 43, 16, 
76, 107, 79, 31, 
47, 57, -2, 84, 
127, 127, 78, 53, 
-98, 91, 35, 40, 
30, 13, 7, -11, 
112, 127, -55, 17, 
0, 119, 1, 39, 
17, 17, 2, -80, 
13, 23, -53, 30, 
47, 73, 17, -69, 
84, -15, -47, -58, 
-66, 4, -62, 55, 
127, 16, 19, -13, 
14, -19, 41, 59, 
70, 93, 32, 80, 
-21, 45, 86, -37, 
21, -73, 18, 72, 
42, 127, 2, -49, 
16, 89, 91, -21, 
-14, -127, 34, 1, 
99, 78, 85, 42, 
47, 127, 88, 42, 
33, 11, 62, 19, 
40, 54, -83, -33, 
51, -24, 127, 70, 
32, -14, 27, 64, 
61, -52, 1, 29, 
81, 46, 92, 92, 
14, 58, -77, 0, 
-45, 27, -101, -44, 
107, -48, 51, 127, 
-17, -14, 0, -16, 
127, 66, 0, 36, 
-6, -44, -28, 41, 
10, -7, 12, -22, 
117, 21, 35, 26, 
-21, -34, -27, 26, 
-6, -41, 54, -17, 
127, 63, 91, 34, 
-88, -85, -112, 30, 
-2, -40, -54, 37, 
-9, 73, 48, 2, 
16, 0, -84, 109, 
14, 2, 21, -5, 
-89, 23, 64, -16, 
65, 60, -8, -8, 
12, -93, 20, 26, 
20, 92, 34, -75, 
127, -51, 67, -39, 
};

const int8_t LineNetwork_layer_2_bias[] = {
21, 3, -42, -20, 7, 19, 17, -17, };


const int8_t LineNetwork_layer_5_weights[] = {
91, -46, 108, -28, 0, -38, -18, -63, 12, -21, 98, 94, 47, -56, -127, 70, -22, -6, -13, -68, -11, 14, -22, -39, -80, 64, 127, -41, -127, 81, 96, -67, 
80, 85, 22, 43, -14, -23, 74, -69, -90, 33, -127, -11, 99, 80, 65, 8, 73, -1, 84, -72, 53, -14, 90, -10, 127, 40, -127, 19, -10, -67, 69, -70, 
-95, 77, 10, -47, 10, -114, 9, 91, 27, 2, -27, 17, -6, 19, 25, -58, 22, -37, -14, -2, -8, -29, -31, 36, -40, 103, 43, 127, 29, 36, -7, 70, 
39, 32, 22, 127, -18, 30, -27, -18, -35, 21, -45, -39, 28, 21, 40, -77, -14, 52, -5, 107, -58, 42, -1, -108, 63, -11, 50, -74, 16, -20, -58, -51, 
-19, 43, 14, -84, 104, 68, -82, 89, -32, -60, 12, -110, -38, -18, 31, 93, -6, -51, -12, 64, 55, 28, 12, 127, 22, -41, -20, -53, -18, 43, -82, 126, 
};

const int8_t LineNetwork_layer_5_bias[] = {
-27, -1, -31, -9, 9, };




LineNetwork::LineNetwork()
	: ModelInterface()
{
	init_buffer(64);
	total_macs = 2373;
	input_channels = 1;
	input_height = 8;
	input_width = 8;
	output_channels = 5;
	output_height = 1;
	output_width = 1;
}

void LineNetwork::forward()
{
	Conv2d<8, 8, 1, 4, 3, 2, int8_t, int8_t, int32_t>(
		output_buffer(), input_buffer(), 
		LineNetwork_layer_0_weights, LineNetwork_layer_0_bias, 80);
	swap_buffer();

	ReLU(	output_buffer(), input_buffer(), 64);
	swap_buffer();

	Conv2d<4, 4, 4, 8, 3, 2, int8_t, int8_t, int32_t>(
		output_buffer(), input_buffer(), 
		LineNetwork_layer_2_weights, LineNetwork_layer_2_bias, 76);
	swap_buffer();

	ReLU(	output_buffer(), input_buffer(), 32);
	swap_buffer();

	Linear<32, 5, int8_t, int8_t, int32_t>(
		output_buffer(), input_buffer(), LineNetwork_layer_5_weights, LineNetwork_layer_5_bias, 66);
	swap_buffer();

	swap_buffer();
}