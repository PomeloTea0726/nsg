//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>
#include <omp.h>
#include <iostream>
#include <load_data.h>

double calculate_recall(std::vector<std::vector<unsigned>>& I, std::vector<std::vector<unsigned>>& gt, int k) {
    assert(I[0].size() >= k);
    assert(gt[0].size() >= k);
    int nq = I.size();
    int total_intersect = 0;

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            for (int t = 0; t < k; t++) {
                if (I[i][j] == gt[i][t]) {
                    total_intersect++;
                    break;
                }
            }
        }
    }
    return static_cast<double>(total_intersect) / (nq * k);
}


int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << argv[0]
              << " dataset nsg_path search_L omp interq_multithread batch_size"
              << std::endl;
    exit(-1);
  }

  float* data_load = NULL;
  unsigned points_num, dim;
  float* query_load = NULL;
  unsigned query_num, query_dim;
  std::vector<std::vector<unsigned>> gt;
  if (strcmp(argv[1], "SIFT1M") == 0) {
    std::cout << "load base vectors..." << std::endl;
    points_num = 1e6;
    load_data_bvecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_base.bvecs", data_load, dim, points_num);
    std::cout << "load query vectors..." << std::endl;
    query_num = 1e4;
    load_data_bvecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_query.bvecs", query_load, query_dim, query_num);
    load_data_ivecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/idx_1M.ivecs", gt, query_num);
  }
  else if (strcmp(argv[1], "SIFT10M") == 0) {
    points_num = 1e7;
    load_data_bvecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_base.bvecs", data_load, dim, points_num);
    query_num = 1e4;
    load_data_bvecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_query.bvecs", query_load, query_dim, query_num);
    load_data_ivecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/idx_10M.ivecs", gt, query_num);
  }
  else if (strcmp(argv[1], "SBERT1M") == 0) {
    points_num = 1e6;
    load_data_SBERT("/mnt/scratch/wenqi/Faiss_experiments/sbert/sbert1M.fvecs", data_load, points_num);
    dim = 384;
    query_num = 1e4;
    load_data_SBERT("/mnt/scratch/wenqi/Faiss_experiments/sbert/query_10K.fvecs", query_load, query_num);
    query_dim = 384;
    load_data_deep_ibin("/mnt/scratch/wenqi/Faiss_experiments/sbert/gt_idx_1M.ibin", gt, query_num);
  }
  else if (strcmp(argv[1], "Deep1M") == 0) {
    points_num = 1e6;
    load_data_deep_fbin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/base.1B.fbin", data_load, points_num);
    dim = 96;
    query_num = 1e4;
    load_data_deep_fbin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/query.public.10K.fbin", query_load, query_num);
    query_dim = 96;
    load_data_deep_ibin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/gt_idx_1M.ibin", gt, query_num);
  }
  else if (strcmp(argv[1], "Deep10M") == 0) {
    points_num = 1e7;
    load_data_deep_fbin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/base.1B.fbin", data_load, points_num);
    dim = 96;
    query_num = 1e4;
    load_data_deep_fbin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/query.public.10K.fbin", query_load, query_num);
    query_dim = 96;
    load_data_deep_ibin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/gt_idx_10M.ibin", gt, query_num);
  }
  else if (strcmp(argv[1], "SPACEV1M") == 0) {
    points_num = 1e6;
    load_data_spacev("/mnt/scratch/wenqi/Faiss_experiments/SPACEV/vectors_all.bin", data_load, points_num);
    dim = 100;
    query_num = 1e4;
    load_data_spacev("/mnt/scratch/wenqi/Faiss_experiments/SPACEV/query_10K.bin", query_load, query_num);
    query_dim = 100;
    load_data_deep_ibin("/mnt/scratch/wenqi/Faiss_experiments/SPACEV/gt_idx_1M.ibin", gt, query_num);
  }
  else if (strcmp(argv[1], "SPACEV10M") == 0) {
    points_num = 1e7;
    load_data_spacev("/mnt/scratch/wenqi/Faiss_experiments/SPACEV/vectors_all.bin", data_load, points_num);
    dim = 100;
    query_num = 1e4;
    load_data_spacev("/mnt/scratch/wenqi/Faiss_experiments/SPACEV/query_10K.bin", query_load, query_num);
    query_dim = 100;
    load_data_deep_ibin("/mnt/scratch/wenqi/Faiss_experiments/SPACEV/gt_idx_10M.ibin", gt, query_num);
  }
  else {
    std::cout << "Unknown dataset" << std::endl;
    exit(-1);
  }
  assert(dim == query_dim);


  unsigned L = (unsigned)atoi(argv[3]);

  int omp = atoi(argv[4]);
  int interq_multithread = atoi(argv[5]);
  int batch_size = atoi(argv[6]);

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build query_load = efanna2e::data_align(query_load,
  // query_num, query_dim);
  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
  std::cout << "load nsg index from " << argv[2] << std::endl;
  index.Load(argv[2]);
  std::cout << "optimize graph" << std::endl;
  index.OptimizeGraph(data_load);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(L);

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
        index.SearchWithOptGraph(query_load + i * dim, L, paras, res[i].data());
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
        index.SearchWithOptGraph(query_load + i * dim, L, paras, res[i].data());
      }
      auto e = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(e - s).count();
      total_time += diff;
      std::cout << diff << " us\n";
    }
  }

  double recall_1, recall_10;
  if (L < 10) {
    recall_1 = calculate_recall(res, gt, 1);
    std::cout << "recall_1: " << recall_1 << std::endl;
  }
  else {
    recall_1 = calculate_recall(res, gt, 1);
    recall_10 = calculate_recall(res, gt, 10);
    std::cout << "recall_1: " << recall_1 << std::endl;
    std::cout << "recall_10: " << recall_10 << std::endl;
  }
  double qps = 1e6 * query_num / total_time;
  std::cout << "qps: " << qps << std::endl;

  return 0;
}
