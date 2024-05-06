//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>
#include <omp.h>
#include <iostream>

#define NUM_ANSWERS 100

double calculate_recall(std::vector<std::vector<unsigned>>& I, std::vector<std::vector<unsigned>>& gt, int k) {
    assert(I[0].size() >= k);
    assert(gt[0].size() >= k);
    int nq = I.size();
    int total_intersect = 0;
    for (int i = 0; i < nq; ++i) {
        std::vector<int> intersection;
        std::set_intersection(I[i].begin(), I[i].begin() + k, gt[i].begin(), gt[i].begin() + k, std::back_inserter(intersection));
        int n_intersect = intersection.size();
        total_intersect += n_intersect;
    }
    return static_cast<double>(total_intersect) / (nq * k);
}

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  // std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

int main(int argc, char** argv) {
  if (argc != 10) {
    std::cout << argv[0]
              << " data_file query_file gt_file nsg_path search_L search_K omp interq_multithread batch_size"
              << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);

  std::ifstream inputGT(argv[3], std::ios::binary);
  std::vector<std::vector<unsigned> > gt(query_num, std::vector<unsigned>(NUM_ANSWERS));

  // std::cout << "load groundtruth" << std::endl;
  for (unsigned i = 0; i < query_num; i++) {
      int t;
      inputGT.read((char *) &t, 4);
      inputGT.read((char *) gt[i].data() , t * 4);
      if (t != NUM_ANSWERS) {
          std::cout << "err";
          return 1;
      }
  }
  inputGT.close();
  // std::cout << "load finished" << std::endl;

  unsigned L = (unsigned)atoi(argv[5]);
  unsigned K = (unsigned)atoi(argv[6]);

  int omp = atoi(argv[7]);
  int interq_multithread = atoi(argv[8]);
  int batch_size = atoi(argv[9]);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build query_load = efanna2e::data_align(query_load,
  // query_num, query_dim);
  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
  index.Load(argv[4]);
  index.OptimizeGraph(data_load);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

  if (query_num % batch_size != 0) {
    std::cout << "query_num must be times of batch_size!" << std::endl;
    exit(1);
  }
  std::cout << "query_num divided into " << query_num / batch_size << " batches" << std::endl;
  std::cout << "time for each batch:" << std::endl;

  int64_t total_time = 0;
  // auto s = std::chrono::high_resolution_clock::now();
  if (omp) {
    for (unsigned j = 0; j < query_num; j += batch_size) {
      auto s = std::chrono::high_resolution_clock::now();
      #pragma omp parallel for num_threads(interq_multithread) schedule(dynamic)
      for (unsigned i = j; i < j + batch_size; i++) {
        index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data());
      }
      auto e = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
      total_time += diff;
      std::cout << diff << " us\n";
    }
  }
  else {
    for (unsigned j = 0; j < query_num; j += batch_size) {
      auto s = std::chrono::high_resolution_clock::now();
      for (unsigned i = j; i < j + batch_size; i++) {
        index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data());
      }
      auto e = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
      total_time += diff;
      std::cout << diff << " us\n";
    }
  }

  double recall = calculate_recall(res, gt, K);
  std::cout << "recall: " << recall << std::endl;
  double qps = 1e6 * query_num / total_time;
  std::cout << "qps: " << qps << std::endl;

  return 0;
}
