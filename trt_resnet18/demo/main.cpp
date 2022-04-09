#include<iostream>
#include<NvInfer.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include"tensorrt.hpp"
#include"opencv2/core/core/hpp"
#include"opencv2/dnn/dnn/hpp"
#include"opencv2/imgcodecs/imgcodecs.hpp"
#include"opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int main()
{
    int cudaNum=0;
    cudaError_t error=cudaGetDeviceCount(&cudaNum);
    if(cudaSuccess != error) return 0;
    if(cudaNum<=0) return 0;
    int idx=0;
    if(cudaNum>1){
        cout<<"please choose the gpu index:"<<endl;
        cin >> idx;
        if(idx >= cudaNum)
            idx=cudaNum -1;
        else if (idx<0)
        {
           idx=0;
        } 
    }
    tensorRT *trt =new tensorRT();
    string enginePath="../engine/resnet18.engine";
    vector<float>weights;
    weights = trt->loadWeight("/home/wending/user/01code/c++/trt_resnet50/weights/conv1.weight.wgt");
    for (int i=0;i<weights.size();i++)
    {
        cout<<"i"<<i<<":"<<weights[i]<<endl;
    }
    //trt->createENG(enginePath);
    trt->Inference_init(enginePath,10);
    string imgPath ="";
    Mat pic = imread(imgPath);
    Mat blob = dnn::blobFromImage(pic,1,Size(256,256),Scalar(127.0,127.0,127.0),false,false);
    float *input=new float[1*3*256*256];
    memcpy(input,blob.data,1*3*256*256*sizeof(float));
    float *output = new float[1*1000]
    trt->doInference(input,1,output);
    for(int i=0;i<1000;i++)
    {
        cout<<i<<":"<<output[i]<<endl;
    }
    delete []input;
    delete []output;   
    cudaSetDevice(idx);
    cudaFree(nullptr);
    cout<<"Hello World!"<<endl;
    return 0;
}