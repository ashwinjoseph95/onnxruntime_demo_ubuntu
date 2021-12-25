#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>
int main()
{
const bool use_cuda = false;
const std::string fn_image = "cat.jpeg";
const std::string fn_model = "super_resolution.onnx";
//environment anf options

Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "SuperResolution");
Ort::SessionOptions session_options;

session_options.SetGraphOptimizationLevel(GraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL));

Ort::Session session(env, fn_model.c_str(), session_options);

Ort::AllocatorWithDefaultOptions allocator;

//model info
const char* input_name = session.GetInputName(0, allocator);
const char* output_name = session.GetOutputName(0, allocator);
auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
auto output_dims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
input_dims[0] = output_dims[0] = 1;
std::vector<const char*> input_names {input_name};  
std::vector<const char*> output_names {output_name};

auto image = imread(fn_image, cv::IMREAD_GRAYSCALE);
cv::resize(image, image, cv::Size(224,224), cv::INTER_LINEAR);
//std::cout<<"image.rows"<<image.rows<<std::endl;
//std::cout<<"image.cols"<<image.cols<<std::endl;

CV_Assert(image.type() == CV_8UC1);

//CV_Assert(image.rows == input_dims[2]&& image.cols==input_dims[3]);
cv::Mat blob = cv::dnn::blobFromImage(image,1.0/255.0);
cv::Mat output(output_dims[2],output_dims[3],CV_32FC1);
std::cout<<"output_dims[2]"<<output_dims[2]<<std::endl;
std::cout<<"output_dims[3]"<<output_dims[3]<<std::endl;
std::cout<<"CV_32FC1"<<CV_32FC1<<std::endl;
auto memory_info = Ort::MemoryInfo::CreateCpu(
   OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
std::vector<Ort::Value> input_tensors, output_tensors;

input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info,blob.ptr<float>(),blob.total(),input_dims.data(), input_dims.size()));
output_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info,output.ptr<float>(),output.total(),output_dims.data(), output_dims.size()));

std::cout<<"input_names.data()"<<input_names.data()<<std::endl;
std::cout<<"output_names.data()"<<output_names.data()<<std::endl;

//inference
    session.Run(Ort::RunOptions{ nullptr }, input_names.data(), input_tensors.data(),1,output_names.data(),output_tensors.data(),1);

    cv::Mat result_ort;
    cv::convertScaleAbs(output, result_ort, 255.0);
    cv::imwrite("result_ort.png",result_ort);

return(0);
}
