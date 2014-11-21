#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sys/socket.h>
#include <arpa/inet.h>   
#include <unistd.h>  
#include <errno.h>

#include "boost/program_options.hpp" 

#include "SENNA_utils.h"
#include "SENNA_Hash.h"
#include "SENNA_Tokenizer.h"
#include "SENNA_POS.h"
#include "SENNA_CHK.h"
#include "SENNA_NER.h"
#include "SENNA_VBS.h"
#include "SENNA_PT0.h"
#include "SENNA_SRL.h"

#include "socket.h"

/* fgets max sizes */
#define MAX_SENTENCE_SIZE 10000000
#define MAX_TARGET_VB_SIZE 256

#define TIMING 1

using namespace std;
namespace po = boost::program_options;

po::variables_map parse_opts( int ac, char** av )
{
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "Produce help message")

        ("task,t", po::value<string>(), "NLP task: pos, chk, ner, srl")
        ("hostname,h", po::value<string>(), "Server IP addr")
        ("portno,p", po::value<int>()->default_value(8080), "Server port (default: 8080)")

		("debug,v", po::value<bool>()->default_value(false), "Turn on all debug") 
		;

	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);    

	if (vm.count("help")) {
		cout << desc << "\n";
		return vm;
	}
	return vm;
}

int main(int argc , char *argv[])
{
    po::variables_map vm = parse_opts(argc, argv);

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
    // SENNA_Hash *psg_left_hash = SENNA_Hash_new(opt_path, "hash/psg-left.lst");
    // SENNA_Hash *psg_right_hash = SENNA_Hash_new(opt_path, "hash/psg-right.lst");

    // JH: Load "weights" + other stuff, most of it unused
    SENNA_POS *pos = SENNA_POS_new(opt_path, "data/pos.dat");
    SENNA_CHK *chk = SENNA_CHK_new(opt_path, "data/chk.dat");
    SENNA_PT0 *pt0 = SENNA_PT0_new(opt_path, "data/pt0.dat");
    SENNA_NER *ner = SENNA_NER_new(opt_path, "data/ner.dat");
    SENNA_VBS *vbs = SENNA_VBS_new(opt_path, "data/vbs.dat");
    SENNA_SRL *srl = SENNA_SRL_new(opt_path, "data/srl.dat");
    // SENNA_PSG *psg = SENNA_PSG_new(opt_path, "data/psg.dat");

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

    bool service = vm.count("hostname");
    int socketfd = -1;
    /* open socket */
    if(service)
        socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
    else
        printf("No hostname provided, default to local DNN.\n");

    // send len over socket
    if(service && socketfd < 0)
        exit(0);
    int len = 0;
    string task = vm["task"].as<string>();
    // send req_type
    int req_type;

    if(task == "pos") {
        req_type = 4;
        len = pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size);
        pos->service = service;
        pos->debug = vm["debug"].as<bool>();
    }
    else if(task == "chk") {
        req_type = 6;
        pos->service = chk->service = service;
        pos->debug = chk->debug = vm["debug"].as<bool>();
        if(pos->service)
            pos->socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
        if(pos->socketfd > 0 && pos->service) {
            int internal_req = 4;
            SOCKET_send(pos->socketfd, (char*)&internal_req, sizeof(int), vm["debug"].as<bool>());
       }
        len = chk->window_size*(chk->ll_word_size+chk->ll_caps_size+chk->ll_posl_size);
    }
    else if(task == "ner") {
        req_type = 5;
        int input_size = ner->ll_word_size+ner->ll_caps_size
                                          +ner->ll_gazl_size+ner->ll_gazm_size+ner->ll_gazo_size+ner->ll_gazp_size;
        len = ner->window_size*input_size;
        ner->service = service;
        ner->debug = vm["debug"].as<bool>();
    }
    else if(task == "srl") {
        req_type = 7;
        pos->service = pt0->service = vbs->service = srl->service = service;
        pos->debug = pt0->debug = vbs->debug = srl->debug = vm["debug"].as<bool>();
        if(pos->service)
            pos->socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
        if(pt0->service)
            pt0->socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
        if(vbs->service)
            vbs->socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
        if(pos->socketfd > 0 && pos->service) {
            int internal_req = 4;
            SOCKET_send(pos->socketfd, (char*)&internal_req, sizeof(int), vm["debug"].as<bool>());
       }

        if(vbs->socketfd > 0 && vbs->service) {
            int internal_req = 9;
            SOCKET_send(vbs->socketfd, (char*)&internal_req, sizeof(int), vm["debug"].as<bool>());
       }
        if(pt0->socketfd > 0 && pt0->service) {
            int internal_req = 8;
            SOCKET_send(pt0->socketfd, (char*)&internal_req, sizeof(int), vm["debug"].as<bool>());
       }
        len = srl->hidden_state1_size;
    }

    if(service) {
        SOCKET_send(socketfd, (char*)&req_type, sizeof(int), vm["debug"].as<bool>());
        if(task == "srl") SOCKET_txsize(socketfd, len);
    }

    while(fgets(sentence, MAX_SENTENCE_SIZE, stdin))
    {
        SENNA_Tokens* tokens = SENNA_Tokenizer_tokenize(tokenizer, sentence);

        if(task != "srl")
          SOCKET_txsize(socketfd, len*tokens->n);

        if(tokens->n == 0)
            continue;

        if(task == "pos"){
           pos_labels = SENNA_POS_forward(pos, tokens->word_idx, tokens->caps_idx, tokens->suff_idx, tokens->n, socketfd);
        }
        else if(task == "chk") {

            SOCKET_txsize(pos->socketfd,
                    tokens->n*(pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)));
 
            pos_labels = SENNA_POS_forward(pos, tokens->word_idx, tokens->caps_idx, tokens->suff_idx, tokens->n, pos->socketfd);
            chk_labels = SENNA_CHK_forward(chk, tokens->word_idx, tokens->caps_idx, pos_labels, tokens->n, socketfd);
        }
        else if(task == "ner")
            ner_labels = SENNA_NER_forward(ner, tokens->word_idx, tokens->caps_idx, tokens->gazl_idx,
                        tokens->gazm_idx, tokens->gazo_idx, tokens->gazp_idx, tokens->n, socketfd);
        else if(task == "srl") {

            SOCKET_txsize(pos->socketfd,
                    tokens->n * (pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)));
             
            SOCKET_txsize(vbs->socketfd,
                    tokens->n * (vbs->window_size*(vbs->ll_word_size+vbs->ll_caps_size+vbs->ll_posl_size)));
 
            SOCKET_txsize(pt0->socketfd,
                    tokens->n * (pt0->window_size*(pt0->ll_word_size+pt0->ll_caps_size+pt0->ll_posl_size)));
 
            pos_labels = SENNA_POS_forward(pos, tokens->word_idx, tokens->caps_idx, tokens->suff_idx, tokens->n, pos->socketfd);
            pt0_labels = SENNA_PT0_forward(pt0, tokens->word_idx, tokens->caps_idx, pos_labels, tokens->n, pt0->socketfd);
            vbs_labels = SENNA_VBS_forward(vbs, tokens->word_idx, tokens->caps_idx, pos_labels, tokens->n, vbs->socketfd);
            n_verbs = 0;
            for(int i = 0; i < tokens->n; i++) {
                vbs_labels[i] = (vbs_labels[i] != vbs_hash_novb_idx);
                n_verbs += vbs_labels[i];
            }

            srl_labels = SENNA_SRL_forward(srl, tokens->word_idx, tokens->caps_idx, pt0_labels, vbs_labels, tokens->n, socketfd);
        }

        if(vm["debug"].as<bool>()) {
            for(int i = 0; i < tokens->n; i++)
            {
                printf("%15s", tokens->words[i]);
                if(task == "pos")
                    printf("\t%10s", SENNA_Hash_key(pos_hash, pos_labels[i]));
                else if(task == "chk")
                    printf("\t%10s", SENNA_Hash_key(chk_hash, chk_labels[i]));
                else if(task == "ner")
                    printf("\t%10s", SENNA_Hash_key(ner_hash, ner_labels[i]));
                else if(task == "srl") {
                    printf("\t%15s", (vbs_labels[i] ? tokens->words[i] : "-"));
                    for(int j = 0; j < n_verbs; j++)
                        printf("\t%10s", SENNA_Hash_key(srl_hash, srl_labels[j][i]));
                }
                printf("\n");
            }
            printf("\n"); /* end of sentence */
        }
    }
#ifdef TIMING
    if(task == "pos")
        cout << "task " << task
            << " size_kb " << (len*sizeof(float))/1024
            << " total_t " << (float)(pos->apptime+pos->dnntime)/1000
            << " app_t " << (float)(pos->apptime/1000)
            << " dnn_t " << (float)(pos->dnntime/1000)
            << " calls " << pos->calls
            << endl;
    else if(task == "chk") {
        cout << "task pos"
            << " size_kb " << (float)(pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)*sizeof(float))/1024
            << " total_t " << (float)(pos->apptime+pos->dnntime)/1000
            << " app_t " << ((float)pos->apptime/1000)
            << " dnn_t " << ((float)pos->dnntime/1000)
            << " calls " << pos->calls
            << endl;
        cout << "task " << task
            << " size_kb " << (float)(len*sizeof(float))/1024
            << " total_t " << (float)(chk->apptime+chk->dnntime)/1000
            << " app_t " << ((float)chk->apptime/1000)
            << " dnn_t " << ((float)chk->dnntime/1000)
            << " calls " << chk->calls
            << endl;
        cout << "task pos+chk"
            << " total_t " << (float)(pos->apptime+pos->dnntime
                             + chk->apptime+chk->dnntime)/1000
            << " app_t " << (float)(pos->apptime
                            + chk->apptime)/1000
            << " dnn_t " << (float)(pos->dnntime
                            + chk->dnntime)/1000
            << " calls " << (float)(pos->calls
                            + chk->calls)
            << endl;
    }
    else if(task == "ner") {
        cout << "task " << task
            << " size_kb " << (float)(len*sizeof(float))/1024
            << " total_t " << (float)(ner->apptime+ner->dnntime)/1000
            << " app_t " << (float)(ner->apptime)/1000
            << " dnn_t " << (float)(ner->dnntime)/1000
            << " calls " << ner->calls
            << endl;
    }
    else if(task == "srl") {
        cout << "task pos"
            << " size_kb " << (pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)*sizeof(float))/1024
            << " total_t " << (pos->apptime+pos->dnntime)/1000
            << " app_t " << pos->apptime/1000
            << " dnn_t " << pos->dnntime/1000
            << " calls " << pos->calls
            << endl;
        cout << "task pt0"
            << " size_kb " << (pt0->window_size*(pt0->ll_word_size+pt0->ll_caps_size+pt0->ll_posl_size)*sizeof(float))/1024
            << " total_t " << (pt0->apptime+pt0->dnntime)/1000
            << " app_t " << pt0->apptime/1000
            << " dnn_t " << pt0->dnntime/1000
            << " calls " << pt0->calls
            << endl;
        cout << "task vbs"
            << " size_kb " << (vbs->window_size*(vbs->ll_word_size+vbs->ll_caps_size+vbs->ll_posl_size)*sizeof(float))/1024
            << " total_t " << (vbs->apptime+vbs->dnntime)/1000
            << " app_t " << vbs->apptime/1000
            << " dnn_t " << vbs->dnntime/1000
            << " calls " << vbs->calls
            << endl;
        cout << "task " << task
            << " size_kb " << (len*sizeof(float))/1024
            << " total_t " << (srl->apptime+srl->dnntime)/1000
            << " app_t " << srl->apptime/1000
            << " dnn_t " << srl->dnntime/1000
            << " calls " << srl->calls
            << endl;
        cout << "task pos+pt0+vbs+srl"
            << " total_t " << (float)(pos->apptime+pos->dnntime
                             + pt0->apptime+pt0->dnntime
                             + vbs->apptime+vbs->dnntime
                             + srl->apptime+srl->dnntime)/1000
            << " app_t " << (float)(pos->apptime
                            + pt0->apptime
                            + vbs->apptime
                            + srl->apptime)/1000
            << " dnn_t " << (float)(pos->dnntime
                            + pt0->dnntime
                            + vbs->dnntime
                            + srl->dnntime)/1000
            << " calls " << (float)(pos->calls
                            + pt0->calls
                            + vbs->calls
                            + srl->calls)
            << endl;
    }
#endif

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

    // SOCKET_close(socketfd);
    
    return 0;
}
