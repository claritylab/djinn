/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "Dnn.h"
#include "caffe/caffe.hpp"
#include "align.h"

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

using namespace boost;
using namespace dnn;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

// default flags (set up for imc)
DEFINE_string(hostname, "localhost", "server host ip");
DEFINE_int32(porno, 8080, "server port");
DEFINE_string(task, "imc", "img task");
DEFINE_string(innet, "input/imc-inputnet.prototxt", "input net");
DEFINE_string(haar, "data/haar.xml", "haar features");
DEFINE_string(flandmark, "data/flandmark.dat", "flandmarks");

int main(int argc, char** argv)
{
    // parse command line
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    shared_ptr<TTransport> socket(new TSocket(FLAGS_hostname, FLAGS_porno));
    shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    DnnClient client(protocol);

    transport->open();

    Work work;
    work.op = FLAGS_task;
    Net<double>* espresso = new Net<double>(FLAGS_innet.c_str());
    const caffe::LayerParameter& in_params = espresso->layers()[0]->layer_param();

    // preprocess face outside of NN for facial recognition before forward pass which loads image(s)
    // $cmt: still tries to do facial recognition if no faces or landmarks found.
    if(work.op == "face")
        if(preprocess(FLAGS_haar, FLAGS_flandmark, in_params.image_data_param().source()) == false)
            exit(0);

    double loss;
    vector<Blob<double>* > img_blobs = espresso->ForwardPrefilled(&loss);
    for(int i = 0; i <img_blobs[0]->count(); ++i)
        work.data.push_back(img_blobs[0]->cpu_data()[i]);

    // receive data
    vector<Blob<double>* > inc_blobs = espresso->output_blobs();

    // forward pass
    vector<double> inc;
    client.fwd(inc, work);

    // check correct
    for(int j = 0; j < inc.size(); ++j)
        cout << "Image: " << j << " class: " << inc[j] << endl;;

    transport->close();
}
