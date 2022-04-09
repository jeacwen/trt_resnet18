#ifndef TENSORRT_HPP
#define TENSORRT_HPP
#include<NvInfer.h>
#include<iostream>
#include<vector>
#include<fstream>
#include<map>
#include<math.h>
using namespace std;


class Logger:public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity,const char *msg) override
    {
        if(severity == Severity::kINFO)
            return;
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr<<"INTERNAL_ERROR:";
            break;
        case Severity::kERROR:
            std::cerr<<"ERROR:";
            break;
        case Severity::kWARNING:
            std::cerr<<"WARNING:";
            break;
        case Severity::kINFO:
            std::cerr<<"INFO:";
            break;
        default:
            std::cerr<<"UNKNOWN:";
            break;
        }
        std::cerr<<msg<<std::endl;
    }
};///class Logger


class tensorRT
{
public:
    tensorRT();
    ~tensorRT();
    ////打印tensor size
    void print_tensor_size(nvinfer1::ITensor*input_tensor,string layerName);
    ////3.创建engine文件
    void createENG(string engPath);
    ////4.加载权重    
    vector<float> loadWeight(string weightPath);
    int loadWeightwd(vector<float>&weights,string weightPath);
    ///定义tensorRT的卷积层
    nvinfer1::ITensor* trt_conv(string inputLayerName,string weightsPath,string biasPath,
                                int output_c,int kernel,int stride,int padding);
    //定义bn层
    nvinfer1::ITensor* trt_batchNorm(string inputLayerName,string filePath);
    //定义激活层
    nvinfer1::ITensor* trt_activation(string inputLayerName,string acti_type);
    //定义池化层
    nvinfer1::ITensor* trt_pool(string inputLayerName,string pool_type,int kernel, int stride,int padding);
    nvinfer1::ITensor* trt_eltwise(string inputLayer1,string inputLayer2,string elttwise_type);
    nvinfer1::ITensor* trt_fc(string inputLayerName,string weightsPath,string biasPath,int output_c);
    ///1.定义log
    Logger m_logger;
    ////2.定义网络
    nvinfer1::INetworkDefinition *m_network;
    ///5.定义存储模型参数的基本结构layers
    map<string,nvinfer1::ITensor*> Layers;
    ///定义数据数据路径
    string rootPath="../weights/";
///////inference/////////////////////////////////////////////////
///1.反序列化engine文件
void Inference_init(const string &engPath,int batchsize);
int inputSize = 3*256*256;
int outputSize = 1000;
////保存输入输出数据
vector<void *> m_binding;
///定义全局上下文推理环境
nvinfer1::IExecutionContext m_context;
//////do inference
void doInference(const float *input,float *output);
///cuda中使用cudaStream获取输入输出数据,定义全局流
cudaStream_t m_cudaStream;
int inputIndex;
int outputIndex;
nvinfer1::ICudaEngine *m_engine 

};////class tensorRT
#endif ////TENSORRT_HPP