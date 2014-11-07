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
#include "SENNA_utils.h"
#include "SENNA_Hash.h"
#include "SENNA_Tokenizer.h"
#include "SENNA_POS.h"
#include "SENNA_CHK.h"
#include "SENNA_NER.h"
#include "SENNA_VBS.h"
#include "SENNA_PT0.h"
#include "SENNA_SRL.h"

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

using namespace boost;
using namespace dnn;

/* fgets max sizes */
#define MAX_SENTENCE_SIZE 1024
#define MAX_TARGET_VB_SIZE 256

// default flags (set up for pos)
DEFINE_bool(service, false, "use service");
DEFINE_bool(debug, true, "debug");
DEFINE_string(hostname, "localhost", "server host ip");
DEFINE_int32(porno, 8080, "server port");
DEFINE_string(task, "pos", "nlp task");

int main(int argc, char** argv)
{
    // parse command line
    google::ParseCommandLineFlags(&argc, &argv, true);

    shared_ptr<TTransport> socket(new TSocket(FLAGS_hostname, FLAGS_porno));
    shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
    DnnClient client(protocol);

    transport->open();

    /* options */
    char *opt_path = NULL;
    int opt_usrtokens = 0;

    /* the real thing */
    char sentence[MAX_SENTENCE_SIZE];
    char target_vb[MAX_TARGET_VB_SIZE];
    int *chk_labels = NULL;
    int *pt0_labels = NULL;
    int *pos_labels = NULL;
    int *ner_labels = NULL;
    int *vbs_labels = NULL;
    int **srl_labels = NULL;
    int *psg_labels = NULL;
    int n_psg_level = 0;
    int is_psg_one_segment = 0;
    int vbs_hash_novb_idx = 22;
    int n_verbs = 0;

    /* inputs */
    SENNA_Hash *word_hash = SENNA_Hash_new(opt_path, "hash/words.lst");
    SENNA_Hash *caps_hash = SENNA_Hash_new(opt_path, "hash/caps.lst");
    SENNA_Hash *suff_hash = SENNA_Hash_new(opt_path, "hash/suffix.lst");
    SENNA_Hash *gazt_hash = SENNA_Hash_new(opt_path, "hash/gazetteer.lst");

    SENNA_Hash *gazl_hash = SENNA_Hash_new_with_admissible_keys(opt_path, "hash/ner.loc.lst", "data/ner.loc.dat");
    SENNA_Hash *gazm_hash = SENNA_Hash_new_with_admissible_keys(opt_path, "hash/ner.msc.lst", "data/ner.msc.dat");
    SENNA_Hash *gazo_hash = SENNA_Hash_new_with_admissible_keys(opt_path, "hash/ner.org.lst", "data/ner.org.dat");
    SENNA_Hash *gazp_hash = SENNA_Hash_new_with_admissible_keys(opt_path, "hash/ner.per.lst", "data/ner.per.dat");

    /* labels */
    SENNA_Hash *pos_hash = SENNA_Hash_new(opt_path, "hash/pos.lst");
    SENNA_Hash *chk_hash = SENNA_Hash_new(opt_path, "hash/chk.lst");
    SENNA_Hash *ner_hash = SENNA_Hash_new(opt_path, "hash/ner.lst");
    SENNA_Hash *vbs_hash = SENNA_Hash_new(opt_path, "hash/vbs.lst");
    SENNA_Hash *srl_hash = SENNA_Hash_new(opt_path, "hash/srl.lst");

    SENNA_POS *pos = SENNA_POS_new(opt_path, "data/pos.dat");
    SENNA_CHK *chk = SENNA_CHK_new(opt_path, "data/chk.dat");
    SENNA_PT0 *pt0 = SENNA_PT0_new(opt_path, "data/pt0.dat");
    SENNA_NER *ner = SENNA_NER_new(opt_path, "data/ner.dat");
    SENNA_VBS *vbs = SENNA_VBS_new(opt_path, "data/vbs.dat");
    SENNA_SRL *srl = SENNA_SRL_new(opt_path, "data/srl.dat");

    /* tokenizer */
    SENNA_Tokenizer *tokenizer = SENNA_Tokenizer_new(word_hash,
                                                     caps_hash,
                                                     suff_hash,
                                                     gazt_hash,
                                                     gazl_hash,
                                                     gazm_hash,
                                                     gazo_hash,
                                                     gazp_hash, 
                                                     opt_usrtokens
                                                     );


    while(fgets(sentence, MAX_SENTENCE_SIZE, stdin))
    {
        SENNA_Tokens* tokens = SENNA_Tokenizer_tokenize(tokenizer, sentence);

        if(tokens->n == 0)
            continue;

        if(FLAGS_task == "pos")
            pos_labels = SENNA_POS_forward(pos, tokens->word_idx, tokens->caps_idx, tokens->suff_idx, tokens->n, client, FLAGS_service);
        else if(FLAGS_task == "chk") {
            pos_labels = SENNA_POS_forward(pos, tokens->word_idx, tokens->caps_idx, tokens->suff_idx, tokens->n, client, FLAGS_service);
            chk_labels = SENNA_CHK_forward(chk, tokens->word_idx, tokens->caps_idx, pos_labels, tokens->n, client, FLAGS_service);
        }
        else if(FLAGS_task == "ner")
            ner_labels = SENNA_NER_forward(ner, tokens->word_idx, tokens->caps_idx, tokens->gazl_idx,
                        tokens->gazm_idx, tokens->gazo_idx, tokens->gazp_idx, tokens->n, client, FLAGS_service);
        else if(FLAGS_task == "srl") {
            pos_labels = SENNA_POS_forward(pos, tokens->word_idx, tokens->caps_idx, tokens->suff_idx, tokens->n, client, FLAGS_service);
            pt0_labels = SENNA_PT0_forward(pt0, tokens->word_idx, tokens->caps_idx, pos_labels, tokens->n, client, FLAGS_service);
            vbs_labels = SENNA_VBS_forward(vbs, tokens->word_idx, tokens->caps_idx, pos_labels, tokens->n, client, FLAGS_service);
            n_verbs = 0;
            for(int i = 0; i < tokens->n; i++) {
                vbs_labels[i] = (vbs_labels[i] != vbs_hash_novb_idx);
                n_verbs += vbs_labels[i];
            }
            srl_labels = SENNA_SRL_forward(srl, tokens->word_idx, tokens->caps_idx, pt0_labels, vbs_labels, tokens->n, client, FLAGS_service);
        }

        if(FLAGS_debug) {
            for(int i = 0; i < tokens->n; i++)
            {
                printf("%15s", tokens->words[i]);
                if(FLAGS_task == "pos")
                    printf("\t%10s", SENNA_Hash_key(pos_hash, pos_labels[i]));
                else if(FLAGS_task == "chk")
                    printf("\t%10s", SENNA_Hash_key(chk_hash, chk_labels[i]));
                else if(FLAGS_task == "ner")
                    printf("\t%10s", SENNA_Hash_key(ner_hash, ner_labels[i]));
                else if(FLAGS_task == "srl") {
                    printf("\t%15s", (vbs_labels[i] ? tokens->words[i] : "-"));
                    for(int j = 0; j < n_verbs; j++)
                        printf("\t%10s", SENNA_Hash_key(srl_hash, srl_labels[j][i]));
                }
                printf("\n");
            }
            printf("\n"); /* end of sentence */
        }
    }

    // clean up
    SENNA_Tokenizer_free(tokenizer);

    SENNA_POS_free(pos);
    SENNA_CHK_free(chk);
    SENNA_PT0_free(pt0);
    SENNA_NER_free(ner);
    SENNA_VBS_free(vbs);
    SENNA_SRL_free(srl);

    SENNA_Hash_free(word_hash);
    SENNA_Hash_free(caps_hash);
    SENNA_Hash_free(suff_hash);
    SENNA_Hash_free(gazt_hash);

    SENNA_Hash_free(gazl_hash);
    SENNA_Hash_free(gazm_hash);
    SENNA_Hash_free(gazo_hash);
    SENNA_Hash_free(gazp_hash);

    SENNA_Hash_free(pos_hash);


    transport->close();
}
