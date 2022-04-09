#include "tensorrt.hpp"
tensorRT::tensorRT()
{
    

}
void tensorRT::print_tensor_size(nvinfer1::ITensor*input_tensor,string layerName)
{
    cout<<layerName<<" ";
    for(int i=0;i<input_tensor->getDimensions().nbDims;i++)
    {
        cout<<input_tensor->getDimensions().d[i]<<" ";
    }
    cout<<endl;
}

vector<float> tensorRT::loadWeight(string weightPath)
{////读二进制
    int size = 0;
    ifstream file(weightPath,ios_base::binary);
    file.read((char*)&size,4);
    char *floatWeights = new char[size*4];
    float *fp = (float*)floatWeights;
    file.read(floatWeights,size*4);
    vector<float>weights(fp,fp+size);
    delete []floatWeights;
    file.close();
    return weights;
}

/*
vector<float> tensorRT::loadWeight(string weightPath)
{
    int size = 0;
    ifstream file(weightPath,ios::in|ios::binary);
    if(!file)
    {
        cout<<"loadWeight error"<<endl;
        exit(-1);
    }
    file.seekg(0,file.end); //基地址为文件结束处，偏移地址为0，于是指针定位在文件结束处
    size = file.tellg()/4;//sp为定位指针，因为它在文件结束处，所以也就是文件的大小
    size = size -1;
    file.seekg(0,file.beg);//基地址为文件头，偏移量为0，于是定位在文件头
    char *floatWeights = new char[size];
    float *fp = (float*)floatWeights;
    file.read(floatWeights,size);
    vector<float>weights(fp,fp+size);
    cout<<"size:"<<size<<endl;
    //for (int i=0;i<size;i++)
    //{
    //    cout<<"i"<<i<<":"<<weights[i]<<endl;
    //}
    delete []floatWeights;
    file.close();
    return weights;
}
*/
void tensorRT::createENG(string engPath)
{
    ////定义输入维度c、h、w
    int input_c=3;
    int input_h=224;
    int input_w=224;
    cout<<"1"<<endl;
    ////定义tensorRT自己的解释器类需要定义数据的初始化
    nvinfer1::IBuilder *builder=nvinfer1::createInferBuilder(m_logger);
    ///builder 开始解释网路
    m_network=builder->createNetwork();
    /////tensor 的形式定义输入输出
    ////data 给input层命名方便检索key-value的结构
    ////nvinfer1::DataType::kFLOAT定义数据类型
    ///nvinfer1::DimsCHW{}将输入数据chw以DimsCHW的形式打包做为参数进行传递
    nvinfer1::ITensor *input = m_network->addInput("data",nvinfer1::DataType::kFLOAT,
                                                    nvinfer1::DimsCHW{static_cast<int>(input_c),
                                                                      static_cast<int>(input_h),
                                                                      static_cast<int>(input_w)});
                                                                
    //将输入层放入layers
    Layers["input"] = input;
    Layers["conv1"] = trt_conv("input","conv1.weight","",64,7,2,3);
    Layers["batchNorm1"] = trt_batchNorm("conv1","bn1");
    Layers["relu1"] = trt_activation("batchNorm1","relu");
    Layers["pool1"] = trt_pool("relu1","max",3,2,1);
    ///renet_type1 layer1.0
    Layers["layer1.0_conv1"] = trt_conv("pool1","layer1.0.conv1.weight","",64,3,1,1);
    Layers["layer1.0_batchNorm1"] = trt_batchNorm("layer1.0_conv1","layer1.0.bn1");
    Layers["layer1.0_relu"] = trt_activation("layer1.0_batchNorm1","relu");

    Layers["layer1.0_conv2"] = trt_conv("layer1.0_relu","layer1.0.conv2.weight","",64,3,1,1);
    Layers["layer1.0_batchNorm2"] = trt_batchNorm("layer1.0_conv2","layer1.0.bn2");

    Layers["layer1.0_sum1"] = trt_eltwise("pool1","layer1.0_batchNorm2","sum");
    Layers["layer1.0_relu1"] = trt_activation("layer1.0_sum1","relu");
    ///renet_type1 layer1.1
    Layers["layer1.1_conv1"] = trt_conv("layer1.0_relu1","layer1.1.conv1.weight","",64,3,1,1);
    Layers["layer1.1_batchNorm1"] = trt_batchNorm("layer1.1_conv1","layer1.1.bn1");
    Layers["layer1.1_relu"] = trt_activation("layer1.1_batchNorm1","relu");

    Layers["layer1.1_conv2"] = trt_conv("layer1.1_relu","layer1.1.conv2.weight","",64,3,1,1);
    Layers["layer1.1_batchNorm2"] = trt_batchNorm("layer1.1_conv2","layer1.1.bn2");

    Layers["layer1.1_sum2"] = trt_eltwise("layer1.0_relu1","layer1.1_batchNorm2","sum");
    Layers["layer1.1_relu2"] = trt_activation("layer1.1_sum2","relu");

    ///resnet_type2 layer2.0
    Layers["layer2.0_conv1"] = trt_conv("layer1.1_relu2","layer2.0.conv1.weight","",128,3,2,1);
    Layers["layer2.0_batchNorm1"] = trt_batchNorm("layer2.0_conv1","layer2.0.bn1");
    Layers["layer2.0_relu"] = trt_activation("layer2.0_batchNorm1","relu");
    Layers["layer2.0_conv2"] = trt_conv("layer2.0_relu","layer2.0.conv2.weight","",128,3,1,1);
    Layers["layer2.0_batchNorm1"] = trt_batchNorm("layer2.0_conv2","layer2.0.bn2"); 

    // layyer2.0.downsample
    Layers["layer2.0_conv1_dowmsample.0"] = trt_conv("layer1.1_relu2","layer2.0.downsample.0.weight","",128,1,2,0);
    Layers["layer2.0_conv1_dowmsample.1"] = trt_batchNorm("layer2.0_conv1_dowmsample.0","layer2.0.downsample.1");
    Layers["layer2.0.eltwise1"] = trt_eltwise("layer2.0_batchNorm1","layer2.0_conv1_dowmsample.1","sum");
    Layers["layer2.0.relu2"] = trt_activation("layer2.0.eltwise1","relu");
    ///renet_type1 layer2.1
    Layers["layer2.1_conv1"] = trt_conv("layer2.0.relu2","layer2.1.conv1.weight","",128,3,1,1);
    Layers["layer2.1_batchNorm1"] = trt_batchNorm("layer2.1_conv1","layer2.1.bn1");
    Layers["layer2.1_relu"] = trt_activation("layer2.1_batchNorm1","relu");
    Layers["layer2.1_conv2"] = trt_conv("layer2.1_relu","layer2.1.conv2.weight","",128,3,1,1);
    Layers["layer2.1_batchNorm2"] = trt_batchNorm("layer2.1_conv2","layer2.1.bn2");

    Layers["layer2.1_sum2"] = trt_eltwise("layer2.0.relu2","layer2.1_batchNorm2","sum");
    Layers["layer2.1_relu2"] = trt_activation("layer2.1_sum2","relu");

    ////layer3.0
    Layers["layer3.0.conv1"] = trt_conv("layer2.1_relu2","layer3.0.conv1.weight","",256,3,2,1);
    Layers["layer3.0.bn1"] = trt_batchNorm("layer3.0.conv1","layer3.0.bn1");
    Layers["layer3.0.relu1"] = trt_activation("layer3.0.bn1","relu");
    Layers["layer3.0.conv2"] = trt_conv("layer3.0.relu1","layer3.0.conv2.weight","",256,3,1,1);
    Layers["layer3.0.bn2"] = trt_batchNorm("layer3.0.conv2","layer3.0.bn2");
    /////layer3.0 downsample
    Layers["layer3.0.downsample.0"] = trt_conv("layer2.1_relu2","layer3.0.downsample.0.weight","",256,1,2,0);
    Layers["layer3.0.downsample.1"] = trt_batchNorm("layer3.0.downsample.0","layer3.0.downsample.1");
    Layers["layer3.0.eltwise1"] = trt_eltwise("layer3.0.downsample.1","layer3.0.bn2","sum");
    Layers["layer3.0.relu2"] = trt_activation("layer3.0.eltwise1","relu");

    ////layer3.1
    Layers["layer3.1.conv1"] = trt_conv("layer3.0.relu2","layer3.1.conv1.weight","",256,3,1,1);
    Layers["layer3.1.bn1"] = trt_batchNorm("layer3.1.conv1","layer3.1.bn1");
    Layers["layer3.1.relu1"] = trt_activation("layer3.1.bn1","relu");
    Layers["layer3.1.conv2"] = trt_conv("layer3.1.relu1","layer3.1.conv2.weight","",256,3,1,1);
    Layers["layer3.1.bn2"] = trt_batchNorm("layer3.1.conv2","layer3.1.bn2");
    Layers["layer3.1.eltwise1"] = trt_eltwise("layer3.1.bn2","layer3.0.relu2","sum");
    Layers["layer3.1.relu2"] = trt_activation("layer3.1.eltwise1","relu");
    ////layer4.0
    Layers["layer4.0.conv1"] = trt_conv("layer3.1.relu2","layer4.0.conv1.weight","",512,3,2,1);
    Layers["layer4.0.bn1"] = trt_batchNorm("layer4.0.conv1","layer4.0.bn1");
    Layers["layer4.0.relu1"] = trt_activation("layer4.0.bn1","relu");
    Layers["layer4.0.conv2"] = trt_conv("layer4.0.relu1","layer4.0.conv2.weight","",512,3,1,1);
    Layers["layer4.0.bn2"] = trt_batchNorm("layer4.0.conv2","layer4.0.bn2"); 
    ////layer4.0 downsample
    Layers["layer4.0.downsample.0"] = trt_conv("layer3.1.relu2","layer4.0.downsample.0.weight","",512,1,2,0);
    Layers["layer4.0.downsample.1"] = trt_batchNorm("layer4.0.downsample.0","layer4.0.downsample.1");
    Layers["layer4.0.eltwise1"] = trt_eltwise("layer4.0.downsample.1","layer4.0.bn2","sum");
    Layers["layer4.0.relu2"] = trt_activation("layer4.0.eltwise1","relu");

    ////layer3.1
    Layers["layer4.1.conv1"] = trt_conv("layer4.0.relu2","layer4.1.conv1.weight","",512,3,1,1);
    Layers["layer4.1.bn1"] = trt_batchNorm("layer4.1.conv1","layer4.1.bn1");
    Layers["layer4.1.relu1"] = trt_activation("layer4.1.bn1","relu");
    Layers["layer4.1.conv2"] = trt_conv("layer4.1.relu1","layer4.1.conv2.weight","",512,3,1,1);
    Layers["layer4.1.bn2"] = trt_batchNorm("layer4.1.conv2","layer4.1.bn2");
    Layers["layer4.1.eltwise1"] = trt_eltwise("layer4.1.bn2","layer4.0.relu2","sum");
    Layers["layer4.1.relu2"] = trt_activation("layer4.1.eltwise1","relu");
    //aveg_pool_size=1
    Layers["avgPool"] = trt_pool("layer4.1.relu2","avg",8,8,0);
    //fc
    Layers["fc"] = trt_fc("avgPool","fc.weight","fc.bias",1000);
    ////设置输出
    Layers["fc"]->setName("output");
    m_network->markOutput(*Layers["fc"]);
    /////构建engine
    //一次推理最大的bacth
    builder->setMaxBatchSize(200);
    //一次占用显存空间
    builder->setMaxWorkspaceSize(1<<30);////1左移30位约1g的空间
    cout<<"engine init ..."<<endl;
    nvinfer1::ICudaEngine *engine=builder->buildCudaEngine(*m_network);
    ////系列化engine
    nvinfer1::IHostMemory *modelStream=engine->serialize();
    ///存储
    ofstream engFile;
    engFile.open(engPath,ios_base::binary);
    engFile.write(static_cast<const char*>(modelStream->data()),modelStream->size());
    engFile.close();
    m_network->destroy();
    engine->destroy();
    builder->destroy();
    modelStream->destroy();

}
nvinfer1::ITensor* tensorRT::trt_conv(string inputLayerName,string weightsPath,string biasPath,
                            int output_c,int kernel,int stride,int padding)
{
    vector<float>weights;
    vector<float> bias;
    weightsPath=rootPath + weightsPath + ".wgt";
    weights = loadWeight(weightsPath);
    if(biasPath !=""){
        biasPath=rootPath + biasPath + ".wgt";
        bias=loadWeight(biasPath);
    }
    unsigned int size = weights.size();
    //init
    nvinfer1::Weights convWeights{nvinfer1::DataType::kFLOAT,nullptr,size};
    nvinfer1::Weights convBias{nvinfer1::DataType::kFLOAT,nullptr,output_c};
    ///add weight data
    float *val_wt = new float[size];
    for(int i=0;i<size;i++)
    {
        val_wt[i] = weights[i];
    }
    convWeights.values=val_wt;

    float *val_bias=new float[output_c];
    for(int i=0;i<output_c;i++)
    {
        val_bias[i]=0.0;
        if(bias.size()!=0)
        {
            val_bias[i] = bias[i];
        }
    }
    convBias.values=val_bias;
    ///数据add到tensorRT的conv中
    nvinfer1::IConvolutionLayer *conv=m_network->addConvolution(*Layers[inputLayerName],output_c,
                                                                nvinfer1::DimsHW{kernel,kernel},convWeights,convBias);
    conv->setStride(nvinfer1::DimsHW{stride,stride});
    conv->setPadding(nvinfer1::DimsHW{padding,padding});
    print_tensor_size(conv->getOutput(0),inputLayerName);
    return conv->getOutput(0);
}
nvinfer1::ITensor* tensorRT::trt_batchNorm(string inputLayerName,string filePath)
{
    vector<float>weights;
    vector<float>bias;
    vector<float>mean;
    vector<float>var;
    string weightsPath=rootPath + filePath + ".weight.wgt";
    string biasPath=rootPath + filePath + ".bias.wgt";
    string meanPath=rootPath + filePath + ".running_mean.wgt";
    string varPath=rootPath + filePath + ".running_var.wgt";
    weights = loadWeight(weightsPath);
    bias = loadWeight(biasPath);
    mean = loadWeight(meanPath);
    var = loadWeight(varPath);
    unsigned int size = bias.size();
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT,nullptr,size};
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT,nullptr,size};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT,nullptr,size};
    ////计算
    vector<float>bn_var;
    for(int i=0;i<size;i++)
    {
        bn_var.push_back(sqrt(var.at(i)+1e-5));
    }
    float *shiftWt = new float[size];
    for(int i=0;i<size;i++)
    {
        shiftWt[i]=bias.at(i)-(mean.at(i)*weights.at(i)/bn_var.at(i));
    }
    shift.values = shiftWt;
    float *scaleWt = new float[size];
    float *powerWt = new float[size];
    for(int i=0;i<size;i++)
    {
        scaleWt[i]=weights.at(i)/bn_var.at(i);
        powerWt[i]=1.0;
    }
    scale.values = scaleWt;
    power.values = powerWt;
    ///数据装填tensorRT IScaleLayer
    nvinfer1::ScaleMode scaleMode = nvinfer1::ScaleMode::kCHANNEL;
    nvinfer1::IScaleLayer *batchNorm = m_network->addScale(*Layers[inputLayerName],scaleMode,shift,scale,power);
    print_tensor_size(batchNorm->getOutput(0),filePath);
    return batchNorm->getOutput(0);
}

nvinfer1::ITensor* tensorRT::trt_activation(string inputLayerName,string acti_type)
{
    nvinfer1::ActivationType Acti_Type;
    if(acti_type == "relu")
    {
        Acti_Type = nvinfer1::ActivationType::kRELU;
    }
    nvinfer1::IActivationLayer *acti=m_network->addActivation(*Layers[inputLayerName],Acti_Type);
    print_tensor_size(acti->getOutput(0),acti_type);
    return acti->getOutput(0);
}
nvinfer1::ITensor* tensorRT::trt_pool(string inputLayerName,string pool_type,int kernel, int stride,int padding)
{
    nvinfer1::PoolingType Pool_Type;
    if(pool_type == "max")
    {
        Pool_Type = nvinfer1::PoolingType::kMAX;
    }
    else if(pool_type == "avg")
    {
        Pool_Type = nvinfer1::PoolingType::kAVERAGE;
    }
    nvinfer1::IPoolingLayer *pool = m_network->addPooling(*Layers[inputLayerName],Pool_Type,nvinfer1::DimsHW{kernel,kernel});
    pool->setStride(nvinfer1::DimsHW{stride,stride});
    //pool->setPrePadding(nvinfer1::DimsHW{padding,padding});
    pool->setPadding(nvinfer1::DimsHW{padding,padding});
    print_tensor_size(pool->getOutput(0),pool_type);
    return pool->getOutput(0);
}
nvinfer1::ITensor* tensorRT::trt_eltwise(string inputLayer1,string inputLayer2,string eltwise_type)
{
    //判断eltwise 类型
    nvinfer1::ElementWiseOperation Eltwise_Type;
    if(eltwise_type == "sum")
    {
        Eltwise_Type=nvinfer1::ElementWiseOperation::kSUM;
    }
    nvinfer1::IElementWiseLayer *eltwise=m_network->addElementWise(*Layers[inputLayer1],*Layers[inputLayer2],Eltwise_Type);
    print_tensor_size(eltwise->getOutput(0),eltwise_type);
    return eltwise->getOutput(0);
}
nvinfer1::ITensor* tensorRT::trt_fc(string inputLayerName,string weightsPath,string biasPath,int output_c)
{
    weightsPath = rootPath + weightsPath + ".wgt";
    vector<float> weights;
    vector<float> bias;
    weights = loadWeight(weightsPath);
    if(biasPath != "")
    {
        biasPath = rootPath + biasPath + ".wgt";
        bias = loadWeight(biasPath);
    }
    unsigned int size = weights.size();
    nvinfer1::Weights fc_wt{nvinfer1::DataType::kFLOAT,nullptr,size};
    nvinfer1::Weights fc_bias{nvinfer1::DataType::kFLOAT,nullptr,output_c};
    float *fc_weights = new float[size];
    for(int i= 0;i<size;i++)
    {
        fc_weights[i]= weights[i];
    }
    fc_wt.values = fc_weights;
    float *fc_bs = new float[output_c];
    for(int i = 0;i<output_c;i++)
    {
        fc_bs[i] = 0.0;
        if(bias.size() != 0)
        {
            fc_bs[i] = bias[i];
        }
    }
    fc_bias.values = fc_bs;
    nvinfer1::IFullyConnectedLayer *fc = m_network->addFullyConnected(*Layers[inputLayerName],output_c,fc_wt,fc_bias);
    return fc->getOutput(0);
}

///////inference/////////////////////////////////////////////////
///1.反序列化engine文件
void tensorRT::Inference_init(const string &engPath,int batchsize)
{
    ifstream cache(engPath,ios::binary);
    cache.seekg(0,ios::end);
    const int engSize = cache.tellg();
    cache.seekg(0,ios::beg);
    void * modelMem=malloc(engSize);
    cache.read((char*)modelMem,engSize);
    cahe.close();
    ///初始化才】runtime、cudaengine engine文件反序列化到显存里面
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(m_logger);
    m_engine = runtime->deserializeCudaEngine(modelMem,engSize,nullptr);
    runtime->destory();
    free(modelMem);
    if(!engine)
    {
        return;
    }
    m_context = m_engine->createExectionContext();
    if(cudaStreamCreate(&cudaStream)!=0) return;
    ////获取输入输出信息维度
    int bindings = m_engine->getNbBindings();
    m_binding.resize(bindings,nullptr);
    ///malloc 显存空间
    inputIndex = m_engine->getBindingIndex("data");
    int flag = cudaMalloc(&m_bindings.at(inputIndex),batchsize*inputSize*sizeof(float));
    if(flag != 0)
    {
        cout<<"malloc error!"<<endl;
        return;
    }
    outputIndex=m_engine->getBindingIndex("output");
    flag = cudaMalloc(&m_bindings.at(outputIndex),batchsize*outputSize*sizeof(float));
    if(flag != 0)
    {
        cout<<"malloc error!"<<endl;
        return;
    }
}

void tensorRT::doInference(const float *input,float *output)
{
    int flag;
    ////内存到显存
    flag = cudaMemcpyAsync(m_binding.at(inputIndex),input,batchsize*inputSize*sizeof(float),cudaMemcpyHostToDevice,m_cudaStream);
    if(flag != 0)
    {
        cout<<"input copy to cuda error!"<<endl;
        return ;
    }
    ////在显存中推理完毕，结果存放在m_bindings中
    m_coontext->enqueue(batchsize,m_bindings.data(),m_cudaStream,nullptr);
    flag = cudaMemcpyAsync(output,m_bindings.at(outputIndex),batchsize*outputSize*sizeof(float),cudaMemcpyDeviceToHost,m_cudaStream);
    if(!=flag)
    {
        cout<<"cuda copy to output error!"<<endl;
        return;
    }
    cudaStreamSynchronize(m_cudaStream);

}
    
tensorRT::~tensorRT()
{
    if(m_context)
    {
        m_context->destroy();
        m_context = nullptr;
    }
    if(m_engine)
    {
        m_engine->destroy();
        m_engine = nullptr;
    }
    for(auto bindings:m_bindjings){
        cudaFree(bindings)
    }
}