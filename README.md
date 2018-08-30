# NCA

精简版caffe-ssd



[build caffe]

g++ caffe/*.cpp caffe/util/*.cpp caffe/layers/*.cpp caffe/proto/*.cc -I./caffe -lopenbls -lprotobuf -fPIC -shared -std=c++11 -o libcaffe.so

export LD_LIBRARY_PATH=/home/wang/NFS/NCA




[build class]

g++ class/00_classification224.cc -Icaffe -L. -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11 -o class224
g++ class/00_classification227.cc -Icaffe -L. -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11 -o class227

g++ class/00_classification-mm.cc -Icaffe -L. -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11 -o class-mobilent



[build mtcnn]

g++ ssd/00_mtcnn.cc -Icaffe -L. -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11 -o mtcnn




[build ssd]

g++ ssd/00_faceboxs.cc -Icaffe -L. -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11 -o faceboxs
g++ ssd/00_mobilenet-ssd.cc -Icaffe -L. -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11 -o mssd
g++ ssd/00_sqz-ssd.cc -Icaffe -L. -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11 -o sqz-ssd
