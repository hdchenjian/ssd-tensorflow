# get int8 quantization model
# python fppb_to_intpb.py ssd.ini

#../tools/pb_to_cambricon/pb_to_cambricon.host --graph=model/ssd300_vgg16_short_mlu.pb --param_file=ssd_model_offline_param_file.txt --core_version="MLU270" --core_num=1 --save_pb=false
#exit
g++ -std=c++11 -I/usr/local/neuware/include offline.cc -o offline `pkg-config --cflags --libs opencv` -L/usr/local/neuware/lib64 -lcnrt
