#include <string>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cnrt.h"

#ifndef CNRT_CHECK
#define CNRT_CHECK(condition)                   \
    do {                                        \
        condition;                              \
    } while (0)

/* cnrtRet_t ret = condition; \ */
/* CHECK_EQ(ret, CNRT_RET_SUCCESS); \ */
#endif

bool LOG_ON = true;

class ClassifierLauncher {
public:
    ClassifierLauncher(std::string offmodel);
    void run_network(cv::Mat &img1);

    cnrtModel_t model_;
    cnrtFunction_t function_;
    cnrtRuntimeContext_t runtime_ctx_;

    cnrtQueue_t queue_;
    cnrtNotifier_t start_notifier_;
    cnrtNotifier_t end_notifier_;
    int64_t* inputSizeS_, *outputSizeS_;
    int inputNum_, outputNum_;
    void** inputMluPtrS;
    void** outputMluPtrS;
    unsigned int in_n_, in_c_, in_h_, in_w_;
    unsigned int out_n_, out_c_, out_h_, out_w_;
    int out_count_;

    ~ClassifierLauncher() {
        cnrtFreeArray(inputMluPtrS, inputNum_);
        cnrtFreeArray(outputMluPtrS, outputNum_);

        cnrtDestroyFunction(function_);
        cnrtDestroyQueue(queue_);
        cnrtDestroyNotifier(&start_notifier_);
        cnrtDestroyNotifier(&end_notifier_);
        cnrtDestroyRuntimeContext(runtime_ctx_);
        cnrtUnloadModel(model_);
    }
};

ClassifierLauncher::ClassifierLauncher(std::string offmodel){
    int dev_id_ = 0;
    cnrtLoadModel(&model_, offmodel.c_str());
    std::string name = "subnet0";

    cnrtDev_t dev;
    CNRT_CHECK(cnrtGetDeviceHandle(&dev, dev_id_));
    CNRT_CHECK(cnrtSetCurrentDevice(dev));
    if (LOG_ON) {
        std::cout << "Init Classifier for device " << dev_id_ << std::endl;
    }

    CNRT_CHECK(cnrtCreateFunction(&(function_)));
    CNRT_CHECK(cnrtExtractFunction(&(function_), model_, name.c_str()));

    CNRT_CHECK(cnrtCreateRuntimeContext(&runtime_ctx_, function_, nullptr));
    cnrtSetRuntimeContextDeviceId(runtime_ctx_, dev_id_);

    if (cnrtInitRuntimeContext(runtime_ctx_, nullptr) != CNRT_RET_SUCCESS) {
        std::cout << "Failed to init runtime context" << std::endl;
        return;
    }

    CNRT_CHECK(cnrtGetInputDataSize(&inputSizeS_, &inputNum_, function_));
    if (LOG_ON) {
        std::cout << "model input num: " << inputNum_ << " input size " << *inputSizeS_ << std::endl;
    }
    CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeS_, &outputNum_, function_));
    if (LOG_ON) {
        std::cout << "model output num: " << outputNum_ << " output size " << *outputSizeS_ << std::endl;
    }

    inputMluPtrS = (void**)malloc(sizeof(void*) * inputNum_);
    for (int k = 0; k < inputNum_; k++){
        cnrtMalloc(&(inputMluPtrS[k]), inputSizeS_[k]);
    }
    outputMluPtrS = (void**)malloc(sizeof(void*) * outputNum_);
    for (int j = 0; j < outputNum_; j++){
        cnrtMalloc(&(outputMluPtrS[j]), outputSizeS_[j]);
    }

    CNRT_CHECK(cnrtCreateQueue(&queue_));
    CNRT_CHECK(cnrtCreateNotifier(&start_notifier_));
    CNRT_CHECK(cnrtCreateNotifier(&end_notifier_));

    int *dimValues = nullptr;
    int dimNum = 0;
    CNRT_CHECK(cnrtGetInputDataShape(&dimValues, &dimNum, 0, function_));
    in_n_ = dimValues[0];
    in_h_ = dimValues[1];
    in_w_ = dimValues[2];
    in_c_ = dimValues[3];
    free(dimValues);
    if (LOG_ON) {
        std::cout << "model input dimNum: " << dimNum << " N: " << in_n_ << " H: " << in_h_
                  << " W: " << in_w_ << " C: " << in_c_ << std::endl;
    }

    CNRT_CHECK(cnrtGetOutputDataShape(&dimValues, &dimNum, 0, function_));
    out_n_ = dimValues[0];
    out_h_ = dimValues[1];
    out_w_ = 1;
    out_c_ = 1;
    out_count_ = out_n_ * out_h_ * out_w_ * out_c_;
    free(dimValues);
    if (LOG_ON) {
        std::cout << "model output dimNum: " << dimNum << " N: " << out_n_ << " H: " << out_h_
                  << " W: " << out_w_ << " C: " << out_c_ << std::endl;
    }
}

void ClassifierLauncher::run_network(cv::Mat &image) {
    int input_size = in_n_ * in_h_ * in_w_ * in_c_;
    void* cpu_data_cast_type = malloc(cnrtDataTypeSize(CNRT_FLOAT16) * input_size);
    if (LOG_ON) {
        std::cout << "image w x h x c: " << image.cols << " x " << image.rows
                  << " x " << image.channels() << std::endl;
    }

    float _R_MEAN = 123.68;
    float _G_MEAN = 116.78;
    float _B_MEAN = 103.94;
    cv::Mat img;
    image.convertTo(img, CV_32FC3);
    cv::resize(img, img, cv::Size(in_w_, in_h_));

    float *cpu_data_ = (float *)malloc(input_size * sizeof(float));
    int channels = img.channels();
    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            cpu_data_[i * img.cols + j] = img.at<cv::Vec3b>(i, j)[0] - _B_MEAN;
        }
    }
    float *cpu_data_ptr = cpu_data_ + img.rows * img.cols;
    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            cpu_data_ptr[i * img.cols + j] = img.at<cv::Vec3b>(i, j)[1] - _G_MEAN;
        }
    }
    cpu_data_ptr = cpu_data_ + img.rows * img.cols * 2;
    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            cpu_data_ptr[i * img.cols + j] =
                img.at<cv::Vec3b>(i, j)[2] - _R_MEAN;
        }
    }
    
    //cv::Mat img_scale(cv::Size(img.cols, img.cols), CV_32FC1);
    //cv::divide(255.0, img, img_scale, -1);
    //void *cpu_data_ = img_scale.data;
    //float img_data[28*28] = {};
    CNRT_CHECK(cnrtCastDataType(cpu_data_, CNRT_FLOAT32,
                                cpu_data_cast_type, CNRT_FLOAT16, input_size, nullptr));
    CNRT_CHECK(cnrtMemcpy(inputMluPtrS[0], cpu_data_cast_type,
                          inputSizeS_[0], CNRT_MEM_TRANS_DIR_HOST2DEV));
    free(cpu_data_cast_type);

    void* param[inputNum_ + outputNum_];
    for (int j = 0; j < inputNum_; j++) {
        param[j] = inputMluPtrS[j];
    }
    for (int j = 0; j < outputNum_; j++) {
        param[inputNum_ + j] = outputMluPtrS[j];
    }
    
    CNRT_CHECK(cnrtInvokeRuntimeContext(runtime_ctx_, param, queue_, nullptr));

    if (cnrtSyncQueue(queue_) == CNRT_RET_SUCCESS) {
        std::cout << "SyncStream success" << std::endl;
    } else {
        std::cout << "SyncStream error" << std::endl;
    }

    void **output_cpu_data_ = (void **)malloc(sizeof(void **) * outputNum_);
    float **outputCpu = (float**)malloc(sizeof(float*) * outputNum_);
    for (int j = 0; j < 1; j++){
        output_cpu_data_[j] = malloc(sizeof(float) * out_count_);
        CNRT_CHECK(cnrtMemcpy(output_cpu_data_[j], outputMluPtrS[j],
                              outputSizeS_[j], CNRT_MEM_TRANS_DIR_DEV2HOST));
        outputCpu[j] = (float*)malloc(cnrtDataTypeSize(CNRT_FLOAT32) * out_count_);
        CNRT_CHECK(cnrtCastDataType(output_cpu_data_[j], CNRT_FLOAT16,
                                    outputCpu[j], CNRT_FLOAT32, out_count_, nullptr));

        float max_score = -1.0F;
        int max_index = -1;
        for(int i = 0; i < out_count_; i++){
            std::cout << ": " << i << " score " << outputCpu[j][i] << std::endl;
            if(outputCpu[j][i] > max_score){
                max_score = outputCpu[j][i];
                max_index = i;
            }
        }
        std::cout << "class: " << max_index << " score " << max_score << std::endl;
        free(output_cpu_data_[j]);
        free(outputCpu[j]);
    }
    free(output_cpu_data_);
    free(outputCpu);
}

void classification_run(std::string offmodel, std::string images_file) {
    ClassifierLauncher* launcher = new ClassifierLauncher(offmodel);
    cv::Mat image_input = cv::imread(images_file);
    launcher->run_network(image_input);
    delete launcher;
}

int main(int argc, char* argv[]) {
    unsigned int real_dev_num;
    cnrtInit(0);
    cnrtGetDeviceCount(&real_dev_num);
    if (real_dev_num == 0) {
        std::cerr << "only have " << real_dev_num << " device(s) " << std::endl;
        cnrtDestroy();
        return -1;
    }

    std::string offmodel = "model/ssd.cambricon";
    std::string images_file = "demo/test.jpg";
    classification_run(offmodel, images_file);

    cnrtDestroy();
    return 0;
}
