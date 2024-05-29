//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>
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

  points_num = 1e6;
  load_data_spacev("/mnt/scratch/wenqi/Faiss_experiments/SPACEV/vectors_all.bin", data_load, points_num);
  dim = 100;
  query_num = 1e4;
  load_data_spacev("/mnt/scratch/wenqi/Faiss_experiments/SPACEV/query_10K.bin", query_load, query_num);
  query_dim = 100;
  load_data_deep_ibin("/mnt/scratch/wenqi/Faiss_experiments/SPACEV/gt_idx_1M.ibin", gt, query_num);

  // write data_load (float) to file nsg_base.txt
  std::ofstream out("nsg_base.txt");
  for (unsigned i = 0; i < points_num; i++) {
    for (unsigned j = 0; j < dim; j++) {
      out << data_load[i * dim + j] << " ";
    }
    out << std::endl;
  }
  out.close();

  // write query_load (float) to file nsg_query.txt
  out.open("nsg_query.txt");
  for (unsigned i = 0; i < query_num; i++) {
    for (unsigned j = 0; j < query_dim; j++) {
      out << query_load[i * query_dim + j] << " ";
    }
    out << std::endl;
  }
  out.close();



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
  for (unsigned i = 0; i < query_num; i++) res[i].resize(10);

  auto s = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < query_num; i++) {
    index.SearchWithOptGraph(query_load + i * dim, 10, paras, res[i].data());
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "search time: " << diff.count() << "\n";

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

  return 0;
}
