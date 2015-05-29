#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sys/socket.h>
#include <arpa/inet.h>   
#include <unistd.h>  
#include <errno.h>
#include <glog/logging.h>

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
#include "tonic.h"

/* fgets max sizes */
#define MAX_TARGET_VB_SIZE 256

using namespace std;
namespace po = boost::program_options;

bool debug;

po::variables_map parse_opts( int ac, char** av )
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message")
    ("common,c", po::value<string>()->default_value("../../common/"), "Directory with configs and weights")
    ("task,t", po::value<string>()->default_value("pos"), "Image task: pos (Part of speech tagging), chk (Chunking), ner (Name Entity Recognition)")
    ("network,n", po::value<string>()->default_value("pos.prototxt"), "Network config file (.prototxt)")
    ("weights,w", po::value<string>()->default_value("pos.caffemodel"), "Pretrained weights (.caffemodel)")
    ("input,i", po::value<string>()->default_value("input/my-input.txt"), "File with text to analyze (.txt)")

    ("djinn,d", po::value<bool>()->default_value(false), "Use DjiNN service?")
    ("hostname,o", po::value<string>()->default_value("localhost"), "Server IP addr")
    ("portno,p", po::value<int>()->default_value(8080), "Server port")

    ("gpu,g", po::value<bool>()->default_value(false), "Use GPU?")
    ("debug,v", po::value<bool>()->default_value(false), "Turn on all debug") 
    ;

	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);    

	if (vm.count("help")) {
		cout << desc << "\n";
    exit(1);
	}
	return vm;
}

int main(int argc , char *argv[])
{
  // google::InitGoogleLogging(argv[0]);
  po::variables_map vm = parse_opts(argc, argv);

  /* SENNA Inits */
  /* options */
  char *opt_path = NULL;
  int opt_usrtokens = 0;

  /* the real thing */
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

  // weights not used
  SENNA_POS *pos = SENNA_POS_new(opt_path, "data/pos.dat");
  SENNA_CHK *chk = SENNA_CHK_new(opt_path, "data/chk.dat");
  SENNA_NER *ner = SENNA_NER_new(opt_path, "data/ner.dat");
  SENNA_PT0 *pt0 = SENNA_PT0_new(opt_path, "data/pt0.dat");
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

  /* Tonic Suite inits */
  TonicSuiteApp app;
  debug = vm["debug"].as<bool>();
  app.task = vm["task"].as<string>();
  app.network = vm["common"].as<string>() + "configs/" + vm["network"].as<string>();
  app.weights = vm["common"].as<string>() + "weights/" + vm["weights"].as<string>();
  app.input = vm["input"].as<string>();

  // DjiNN service or local?
  app.djinn = vm["djinn"].as<bool>();
  app.gpu = vm["gpu"].as<bool>();

  if(app.djinn) {
    app.hostname = vm["hostname"].as<string>();
    app.portno = vm["portno"].as<int>();
    app.socketfd = CLIENT_init(app.hostname.c_str(), app.portno, debug);
    if(app.socketfd < 0)
      exit(0);
  }
  else {
    app.net = new Net<float>(app.network);
    app.net->CopyTrainedLayersFrom(app.weights);
  }

  strcpy(app.pl.req_name, app.task.c_str());
  if(app.task == "pos")
    app.pl.size =
      pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size);
  else if(app.task == "chk") {
    app.pl.size =
      chk->window_size*(chk->ll_word_size+chk->ll_caps_size+chk->ll_posl_size);
  }
  else if(app.task == "ner") {
    int input_size = ner->ll_word_size+ner->ll_caps_size
      +ner->ll_gazl_size+ner->ll_gazm_size+ner->ll_gazo_size+ner->ll_gazp_size;
    app.pl.size = ner->window_size*input_size;
  }
  // else if(task == "srl") {
  //   req_type = 7;
  //   pos->service = pt0->service = vbs->service = srl->service = service;
  //   pos->debug = pt0->debug = vbs->debug = srl->debug = vm["debug"].as<bool>();
  //   if(pos->service)
  //     pos->socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
  //   if(pt0->service)
  //     pt0->socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
  //   if(vbs->service)
  //     vbs->socketfd = CLIENT_init(vm["hostname"].as<string>().c_str(), vm["portno"].as<int>(), vm["debug"].as<bool>());
  //   if(pos->socketfd > 0 && pos->service) {
  //     int internal_req = 4;
  //     SOCKET_send(pos->socketfd, (char*)&internal_req, sizeof(int), vm["debug"].as<bool>());
  //   }
  //
  //   if(vbs->socketfd > 0 && vbs->service) {
  //     int internal_req = 9;
  //     SOCKET_send(vbs->socketfd, (char*)&internal_req, sizeof(int), vm["debug"].as<bool>());
  //   }
  //   if(pt0->socketfd > 0 && pt0->service) {
  //     int internal_req = 8;
  //     SOCKET_send(pt0->socketfd, (char*)&internal_req, sizeof(int), vm["debug"].as<bool>());
  //   }
  //   len = srl->hidden_state1_size;
  // }

  // read input file
  ifstream file (app.input.c_str());
  string str;
  string text;
  while(getline(file, str))
    text += str;

  // tokenize
  SENNA_Tokens* tokens = SENNA_Tokenizer_tokenize(tokenizer, text.c_str());
  app.pl.num = tokens->n;

  if(app.pl.num == 0)
    LOG(FATAL) << app.input << " empty or no tokens found.";

  // if(task == "srl") SOCKET_txsize(socketfd, len);

  if(app.task == "pos") {
    if(app.djinn) {
      // send app
      SOCKET_send(app.socketfd, (char*)&app.pl.req_name, MAX_REQ_SIZE, debug);
      // send len
      SOCKET_txsize(app.socketfd, app.pl.num * app.pl.size);
    }
    else
      reshape(app.net, app.pl.num * app.pl.size);

    pos_labels = SENNA_POS_forward(pos,
        tokens->word_idx,
        tokens->caps_idx,
        tokens->suff_idx,
        app);
  }
  else if(app.task == "chk") {
    // chk needs internal pos
    TonicSuiteApp pos_app = app;
    pos_app.task = "pos";
    pos_app.network = vm["common"].as<string>() + "configs/" + "pos.prototxt";
    pos_app.weights = vm["common"].as<string>() + "weights/" + "pos.caffemodel";

    if(!pos_app.djinn) {
      pos_app.net = new Net<float>(pos_app.network);
      pos_app.net->CopyTrainedLayersFrom(pos_app.weights);
    }
    strcpy(pos_app.pl.req_name, pos_app.task.c_str());
    pos_app.pl.size =
      pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size);

    // send pos app
    if(pos_app.djinn) {
      pos_app.socketfd = CLIENT_init(pos_app.hostname.c_str(), pos_app.portno, debug);
      SOCKET_send(pos_app.socketfd, (char*)&pos_app.pl.req_name, MAX_REQ_SIZE, debug);
      // send len
      SOCKET_txsize(pos_app.socketfd, pos_app.pl.num * pos_app.pl.size);
    }
    else
      reshape(pos_app.net, pos_app.pl.num * pos_app.pl.size);

    pos_labels = SENNA_POS_forward(pos,
        tokens->word_idx,
        tokens->caps_idx,
        tokens->suff_idx,
        pos_app);

    SOCKET_close(pos_app.socketfd, debug);
    // chk foward pass
    if(app.djinn) {
      SOCKET_send(app.socketfd, (char*)&app.pl.req_name, MAX_REQ_SIZE, debug);
      // send len
      SOCKET_txsize(app.socketfd, app.pl.num * app.pl.size);
    }
    else {
      free(pos_app.net);
      reshape(app.net, app.pl.num * app.pl.size);
    }

    chk_labels = SENNA_CHK_forward(chk,
        tokens->word_idx,
        tokens->caps_idx,
        pos_labels,
        app);
  }
  else if(app.task == "ner") {
    if(app.djinn) {
      // send app
      SOCKET_send(app.socketfd, (char*)&app.pl.req_name, MAX_REQ_SIZE, debug);
      // send len
      SOCKET_txsize(app.socketfd, app.pl.num * app.pl.size);
    }
    else
      reshape(app.net, app.pl.num * app.pl.size);

    ner_labels = SENNA_NER_forward(ner,
        tokens->word_idx,
        tokens->caps_idx,
        tokens->gazl_idx,
        tokens->gazm_idx,
        tokens->gazo_idx,
        tokens->gazp_idx,
        app);
  }
  // else if(task == "srl") {
  //
  //   SOCKET_txsize(pos->socketfd,
  //       tokens->n * (pos->window_size*(pos->ll_word_size+pos->ll_caps_size+pos->ll_suff_size)));
  //
  //   SOCKET_txsize(vbs->socketfd,
  //       tokens->n * (vbs->window_size*(vbs->ll_word_size+vbs->ll_caps_size+vbs->ll_posl_size)));
  //
  //   SOCKET_txsize(pt0->socketfd,
  //       tokens->n * (pt0->window_size*(pt0->ll_word_size+pt0->ll_caps_size+pt0->ll_posl_size)));
  //
  //   pos_labels = SENNA_POS_forward(pos, tokens->word_idx, tokens->caps_idx, tokens->suff_idx, tokens->n, pos->socketfd);
  //   pt0_labels = SENNA_PT0_forward(pt0, tokens->word_idx, tokens->caps_idx, pos_labels, tokens->n, pt0->socketfd);
  //   vbs_labels = SENNA_VBS_forward(vbs, tokens->word_idx, tokens->caps_idx, pos_labels, tokens->n, vbs->socketfd);
  //   n_verbs = 0;
  //   for(int i = 0; i < tokens->n; i++) {
  //     vbs_labels[i] = (vbs_labels[i] != vbs_hash_novb_idx);
  //     n_verbs += vbs_labels[i];
  //   }
  //
  //   std::cout<<"word index is " << tokens->word_idx << std::endl;
  //   std::cout<<"pt0 labels is " << pt0_labels << std::endl;
  //   srl_labels = SENNA_SRL_forward(srl, tokens->word_idx, tokens->caps_idx, pt0_labels, vbs_labels, tokens->n, socketfd);
  // }

  for(int i = 0; i < tokens->n; i++)
  {
    printf("%15s", tokens->words[i]);
    if(app.task == "pos")
      printf("\t%10s", SENNA_Hash_key(pos_hash, pos_labels[i]));
    else if(app.task == "chk")
      printf("\t%10s", SENNA_Hash_key(chk_hash, chk_labels[i]));
    else if(app.task == "ner")
      printf("\t%10s", SENNA_Hash_key(ner_hash, ner_labels[i]));
    else if(app.task == "srl") {
      printf("\t%15s", (vbs_labels[i] ? tokens->words[i] : "-"));
      for(int j = 0; j < n_verbs; j++)
        printf("\t%10s", SENNA_Hash_key(srl_hash, srl_labels[j][i]));
    }
    printf("\n");
  }
  // end of sentence
  printf("\n"); 

  // clean up
  SENNA_Tokenizer_free(tokenizer);

  SENNA_POS_free(pos);
  SENNA_CHK_free(chk);
  SENNA_NER_free(ner);
  // SENNA_PT0_free(pt0);
  // SENNA_VBS_free(vbs);
  // SENNA_SRL_free(srl);

  SENNA_Hash_free(word_hash);
  SENNA_Hash_free(caps_hash);
  SENNA_Hash_free(suff_hash);
  SENNA_Hash_free(gazt_hash);

  SENNA_Hash_free(gazl_hash);
  SENNA_Hash_free(gazm_hash);
  SENNA_Hash_free(gazo_hash);
  SENNA_Hash_free(gazp_hash);

  SENNA_Hash_free(pos_hash);

  SOCKET_close(app.socketfd, debug);

  if(!app.djinn)
    free(app.net);

  return 0;
}
