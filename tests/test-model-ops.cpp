#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <map>
#include <utility> // for pair inclusion

// Pratik, adding gguf-split header
#include "llama.h"
#include "common.h"

#include <ggml-quants.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <stdio.h>
#include <string.h>
#include <climits>
#include <stdexcept>

#if defined(_WIN32)
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

// Pratik

#define MAX_NARGS 2

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

static float frand(void) {
    return (float)rand()/(float)RAND_MAX;
}

static struct ggml_tensor * get_random_tensor(
    struct ggml_context * ctx0, int ndims, int64_t ne[], float fmin, float fmax
) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)result->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)result->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    }

    return result;
}

static void printTensor(struct ggml_tensor * t, const std::string str)
{

    printf("\n%s \n", str.c_str());
    auto dim1 = t->ne[0];
    auto dim2 = t->ne[1];
    auto dim3 = t->ne[2];
    auto dim4 = t->ne[3];

    printf("dim1 : %lld dim2 : %lld dim3 : %lld dim4 : %lld\n", dim1, dim2, dim3, dim4);

    for(int b=0; b<dim4; ++b) // along the batch dim
    {
        for(int ch=0; ch<dim3; ++ch) // along the channel dim
        {
            for(int h=0; h<dim2; ++h) // along the height
            {
                const int pitch3 = dim3*dim2*(dim1);
                const int pitch2 = dim2*(dim1);
                const int pitch1 = (dim1);

                for(int w=0; w<dim1; ++w) // along the width
                {
                    if(t->type == GGML_TYPE_F32)
                    {
                        float val = ((float *)t->data)[b*pitch3 + ch*pitch2 + h*pitch1 + w];
                        printf("%f ", val);
                    }
                    else if(t->type == GGML_TYPE_Q8_0)
                    {
                        /*
                            typedef struct {
                            ggml_half d;       // delta
                            int8_t  qs[QK8_0]; // quants
                            } block_q8_0;

                            based on above defination Q8_0 are stored as int8_t
                        */
                        int8_t val = ((int8_t *)t->data)[b*pitch3 + ch*pitch2 + h*pitch1 + w];
                        printf("%d ", val);    
                    }
                    else
                    {
                        printf("Unsupported data type\n");
                        return;
                    }
                }
                printf("\n");
            }
        }
    }
    printf("\n");
    return;
}


static void read_tensors(struct gguf_context * ctx_gguf,struct ggml_context * ctx_meta, std::map<std::string, struct ggml_tensor *> &extracted_tensors )
{
    for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) 
    {
        // read tensor meta and prepare buffer
        const char * t_name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
        extracted_tensors[t_name] = t;
    }
}

static void gguf_parser(std::string input, std::map<std::string, struct ggml_tensor *> &extracted_tensors) {
    struct ggml_context * ctx_meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_meta,
    };


    std::ifstream f_input(input.c_str(), std::ios::binary);
    if (!f_input.is_open()) {
        fprintf(stderr, "%s:  failed to open input GGUF from %s\n", __func__, input.c_str());
        exit(EXIT_FAILURE);
    }

    auto * ctx_gguf = gguf_init_from_file(input.c_str(), params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s:  failed to load input GGUF from %s\n", __func__, input.c_str());
        exit(EXIT_FAILURE);
    }

    // prepare the strategy
    read_tensors(ctx_gguf, ctx_meta, extracted_tensors);
    
    // done, clean up
    gguf_free(ctx_gguf);
    f_input.close();
}

/*
    Useful for layer level debugs, for every node create a subgraph consisting of single node
    Infer and print tensor for quick verification of layer level output
*/
static void create_and_execute_ggml_sub_graph(struct ggml_tensor * t, struct ggml_context * ctx,  const char * t_name)
{
    struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(ge, t);
    ggml_graph_reset(ge);

    ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    printTensor(t, t_name);
    return;
}

/*utility function to create ggml tensor type from float data buffer*/
static void create_ftensor_from_block(float * ptr_f_values, struct ggml_tensor * t_float_input)
{
    int itr = 0;
    auto dim1 = t_float_input->ne[0];
    auto dim2 = t_float_input->ne[1];
    auto dim3 = t_float_input->ne[2];
    auto dim4 = t_float_input->ne[3];

    for(int b=0; b<dim4; ++b) // along the batch dim
    {
        for(int ch=0; ch<dim3; ++ch) // along the channel dim
        {
            for(int h=0; h<dim2; ++h) // along the height
            {
                const int pitch3 = dim3*dim2*(dim1);
                const int pitch2 = dim2*(dim1);
                const int pitch1 = (dim1);

                for(int w=0; w<dim1; ++w) // along the width
                {
                    ((float *)t_float_input->data)[b*pitch3 + ch*pitch2 + h*pitch1 + w] = ptr_f_values[itr++];  
                }
            }
        }
    }

    return;
}

template<typename BlockType>
static void create_qtensor_from_block_var(BlockType * ptr_t_quant, struct ggml_tensor * t_quant_input, const int nb, const int qk)
{
    // extract the quantized values to single buffer
    int itr = 0;
    int8_t * qvalues = (int8_t *) malloc(nb * qk * sizeof(int8_t));
    for(int i=0; i<nb; ++i)//for every quantized block
    {   
        for(int j=0; j<qk; ++j) // for every element in q block
        {
            qvalues[itr++] = ptr_t_quant[i].qs[j];
        }
    }

    /*setting to 0, so we can read from start*/
    itr = 0;
    auto dim1 = t_quant_input->ne[0];
    auto dim2 = t_quant_input->ne[1];
    auto dim3 = t_quant_input->ne[2];
    auto dim4 = t_quant_input->ne[3];

    /* Iterate through the tensor and store the values */
    for(int b=0; b<dim4; ++b) /*along the batch dim*/
    {
        for(int ch=0; ch<dim3; ++ch) /*along the channel dim*/
        {
            for(int h=0; h<dim2; ++h) /*along the height dim*/
            {
                const int pitch3 = dim3*dim2*(dim1);
                const int pitch2 = dim2*(dim1);
                const int pitch1 = (dim1);

                for(int w=0; w<dim1; ++w) /*along the width dim*/
                {
                    ((int8_t *)t_quant_input->data)[b*pitch3 + ch*pitch2 + h*pitch1 + w] = qvalues[itr++];  
                }
            }
        }
    }

    return;
}

static void update_block_from_qtensor(block_q8_0 * ptr_t_quant, struct ggml_tensor * t_quant_input, const int qk)
{

    /* ptr for block, set to first block*/
    int block_id = 0;
    // once the q-values are extracted, store them to t_qunat_input tensor
    auto dim1 = t_quant_input->ne[0];
    auto dim2 = t_quant_input->ne[1];
    auto dim3 = t_quant_input->ne[2];
    auto dim4 = t_quant_input->ne[3];

    for(int b=0; b<dim4; ++b) // along the batch dim
    {
        for(int ch=0; ch<dim3; ++ch) // along the channel dim
        {
            for(int h=0; h<dim2; ++h) // along the height
            {

                const int pitch3 = dim3*dim2*(dim1);
                const int pitch2 = dim2*(dim1);
                const int pitch1 = (dim1);

                for(int w=0; w<dim1; w+=qk) // along the width
                {
                    for(int j=0; j<qk; ++j) // for every element in q block
                    {
                        ptr_t_quant[block_id].qs[j] = ((int8_t *)t_quant_input->data)[b*pitch3 + ch*pitch2 + h*pitch1 + w+j];  
                    }
                    block_id++;
                }
            }
        }
    }
    return;
}

static struct ggml_tensor * apply_row_wise_quantization(struct ggml_context * ctx, struct ggml_tensor * t_float_input)
{
    auto dim1 = t_float_input->ne[0];
    auto dim2 = t_float_input->ne[1];
    auto dim3 = t_float_input->ne[2];
    auto dim4 = t_float_input->ne[3];

    /*
        t_quant will be return by function, which holds quantized values.
        Qunatization is scalar operation so, it will not effect the size of tensor
    */
    int64_t ne[4] = {dim1, dim2, dim3, dim4};
    struct ggml_tensor * t_quant = ggml_new_tensor(ctx, GGML_TYPE_Q8_0, 4, ne);

    const int qk = dim1;

    /*(dim1 / qk) will be always 1*/
    const int nb = 1;
    const int rows = dim2;

    /* Allocate the total memory needed for the block*/
    block_q8_0_var * ptr_t_quant = (block_q8_0_var *) malloc(rows * nb * sizeof(block_q8_0_var));
    if (ptr_t_quant == NULL) {
        fprintf(stderr, "Memory allocation for array of structures failed\n");
        return t_quant;
    }

    for (int i = 0; i < rows * nb; i++) {
        ptr_t_quant[i].qs = (int8_t *) malloc(dim1 * sizeof(int8_t));
        if (ptr_t_quant[i].qs == NULL) {
            fprintf(stderr, "Memory allocation for qs failed at index %d\n", i);
            // Cleanup previously allocated memory
            for (int j = 0; j < i; j++) {
                free(ptr_t_quant[j].qs);
            }
            free(ptr_t_quant);
            return t_quant;
        }
    }

    quantize_q8_0_row_wise(static_cast<const float*>(t_float_input->data), ptr_t_quant, dim2, dim1, nullptr);

    create_qtensor_from_block_var<block_q8_0_var>(ptr_t_quant, t_quant, (rows * nb), qk);

#ifdef DEBUG
    printf("Printing q tensor after conversion\n");
    printTensor(t_quant, "t_quant");
#endif

    return t_quant;
}

/*
    Graph Structure : 
    [Input -> Quant -> Act1 -> Weight1(INT8, MatMul) -> Act2 -> 
    Weight2(INT8, MatMul) -> Act3 - > Dequant -> Output]

    Assumptions/Retrictions if any :
    Input of size 64x64x1x1 (2D) with row-wise quantization is validated

    Input : file path for to gguf file contains weights
    Ouput : void

    To Do if any : 
     
*/
static void test_case_custom(std::string input_file)
{
    std::map<std::string, struct ggml_tensor *> extracted_tensors;
    gguf_parser(input_file, extracted_tensors);

    try
    {
        gguf_parser(input_file, extracted_tensors);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing input file: " << e.what() << '\n';
        return;
    }

    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    /* input tensor dims*/
    int64_t ne1[4] = {64, 1, 1, 1};

    /* Create required input and output tensors for the given test,
        t_<tensor_name> nomenclature followed for ggml_tensor names
    */
    struct ggml_tensor * t_float_input = get_random_tensor(ctx, 2, ne1, -1, 10);
    ggml_set_param(ctx, t_float_input);
    
    struct ggml_tensor * t_quant_input = ggml_new_tensor(ctx, GGML_TYPE_Q8_0, 4, ne1);
    
    /*
        Apply row wise qunatization on matmul weights
    */
    struct ggml_tensor * t_matmul_1 = apply_row_wise_quantization(ctx, extracted_tensors["matmul1.weight"]);
    struct ggml_tensor * t_matmul_2 = apply_row_wise_quantization(ctx, extracted_tensors["matmul2.weight"]);

    /*
        ptr_t_quant :

        This will hold info about each blocks d value and quantized data for each block
        ptr_t_quant[i] will represent ith block,
        ptr_t_quant[i].d will represent d(scale value)
        ptr_t_quant[i].qs[32] 32 value array
    */
    const int qk = QK8_0;
    const int nb =  (t_float_input->ne[0]/ qk) * t_float_input->ne[1];
    /* ptr_t_quant will hold the base ptr to qunatized values of input tenosr, follows default quantization*/
    block_q8_0 * ptr_t_quant = (block_q8_0 *) malloc(nb * sizeof(block_q8_0));
    /* ptr_t_quant_output hold scales, aka 'd' param that will be used to dequantize ouput tensor */
    block_q8_0 * ptr_t_quant_output = (block_q8_0 *) malloc(nb * sizeof(block_q8_0));
    /* Quantized the input tensor and stror the value t_quant_input tensor*/
    quantize_q8_0(static_cast<const float*>(t_float_input->data), ptr_t_quant, t_float_input->ne[1], t_float_input->ne[0], nullptr);
    create_qtensor_from_block_var<block_q8_0>(ptr_t_quant, t_quant_input, nb, qk);

    #ifdef DEBUG
    printTensor(t_quant_input, "t_quant_input");
    #endif

    ggml_set_param(ctx, t_quant_input);
    
    /* Create graph and execute ggml graph in Q8_0 flow*/
    struct ggml_tensor * act_1 = ggml_relu(ctx, t_quant_input);
    struct ggml_tensor * matMul_1 = ggml_gemm(ctx, act_1, t_matmul_1);
    struct ggml_tensor * act_2  = ggml_relu(ctx, matMul_1);
    struct ggml_tensor * matMul_2 = ggml_gemm(ctx, act_2, t_matmul_2);
    struct ggml_tensor * output_tensor  = ggml_relu(ctx, matMul_2);


    create_and_execute_ggml_sub_graph(output_tensor, ctx, "output_tensor");

    /*
        Reader Note :
        The above graph can be created and infered using ggml_mul_mat, with correct dimentional
        inputs, the flow reamins same.
        
        struct ggml_tensor * act_1 = ggml_relu(ctx, t_quant_input);
        struct ggml_tensor * matMul_1 = ggml_mul_mat(ctx, act_1, t_matmul_1);
        struct ggml_tensor * act_2  = ggml_relu(ctx, matMul_1);
        struct ggml_tensor * matMul_2 = ggml_mul_mat(ctx, act_2, t_matmul_2);
        struct ggml_tensor * output_tensor  = ggml_relu(ctx, matMul_2);

        // Create and infer above graph
        struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(ge, output_tensor);
        ggml_graph_reset(ge);

        ggml_graph_compute_with_ctx(ctx, ge, (n_threads) 1);

    */

    /*
        Dequantization :
        Run the float pass of the model to figure out the output max float values.
        Qunatize the float values to Q8_0 this will yeild d(scales)
        Utilize this scales for ouput layer dequantization, same can be extended for 
        layer level dequnatization however, q_i * q_j = q_k, needs diff implementation than current
    */

    struct ggml_tensor * F_act_1 = ggml_relu(ctx, t_float_input);
    struct ggml_tensor * F_matMul_1 = ggml_gemm(ctx, F_act_1, extracted_tensors["matmul1.weight"]);
    struct ggml_tensor * F_act_2  = ggml_relu(ctx, F_matMul_1);
    struct ggml_tensor * F_matMul_2 = ggml_gemm(ctx, F_act_2, extracted_tensors["matmul2.weight"]);
    struct ggml_tensor * F_output_tensor  = ggml_relu(ctx, F_matMul_2);
    
    create_and_execute_ggml_sub_graph(F_output_tensor, ctx, "F_output_tensor");

    quantize_q8_0(static_cast<const float*>(F_output_tensor->data), ptr_t_quant_output, F_output_tensor->ne[1], F_output_tensor->ne[0], nullptr);
    /*
        Now that we have we scale(d) for each block of float output, lets use this to dequantize 
        output_tensor(Q8_0 to F32)
        For that we need to override the values in qs field of ptr_t_qunat_ouput, so this can be passed 
        to dequantize_row_q8_0 function.
     */

    const int total_nb_output =  (output_tensor->ne[0]/ qk) * output_tensor->ne[1];
    update_block_from_qtensor(ptr_t_quant_output, output_tensor, qk);

    /*
        now that we have updated ptr_t_qunat_output with acutal values of qunat pass,
        lets call the dequantizer 
    */

    float * ptr_t_float_output = (float *) malloc(total_nb_output * qk * sizeof(float));
    dequantize_row_q8_0(ptr_t_quant_output, ptr_t_float_output, output_tensor->ne[0]);

    /*
        At last lets create a float tensor and print the final result of graph,
        We can override the same F_output_tensor to hold dequantized values
     */

    create_ftensor_from_block(ptr_t_float_output, F_output_tensor);
    printTensor(F_output_tensor, "F_output_tensor");

    ggml_free(ctx);
    return;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <test_case>\n";
        std::cerr << "Available test cases: test_case_1, test_case_2\n";
        return 1;
    }

    std::string testCase = argv[2];
    if (testCase == "test_case_1")
    {
        std::cerr << "Test case 1 is currently not implemented.\n";
        return 1;
    } 
    else if (testCase == "test_case_custom")
    {
        test_case_custom(argv[1]);
    } 
    else 
    {
        std::cerr << "Invalid choice. Available choices are: test_case_1, test_case_2\n";
        return 1;
    }

    return 0;
}

