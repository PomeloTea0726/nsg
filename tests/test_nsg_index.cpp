//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <load_data.h>
#include <cstring>


int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << argv[0] << " dataset nn_graph_path L R C save_graph_file"
              << std::endl;
    exit(-1);
  }

  float* data_load = NULL;
  unsigned points_num, dim;
  if (strcmp(argv[1], "sift1m") == 0) {
    points_num = 1e6;
    load_data_bvecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_base.bvecs", data_load, dim, points_num);
  }
  else if (strcmp(argv[1], "sift10m") == 0) {
    points_num = 1e7;
    load_data_bvecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_base.bvecs", data_load, dim, points_num);
  }
  else if (strcmp(argv[1], "SBERT1M") == 0) {
    points_num = 1e6;
    load_data_SBERT("/mnt/scratch/wenqi/Faiss_experiments/sbert/sbert1M.fvecs", data_load, points_num);
    dim = 384;
  }
  else {
    std::cout << "Unknown dataset" << std::endl;
    exit(-1);
  }

  std::string nn_graph_path(argv[2]);
  unsigned L = (unsigned)atoi(argv[3]);
  unsigned R = (unsigned)atoi(argv[4]);
  unsigned C = (unsigned)atoi(argv[5]);

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build
  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);

  auto s = std::chrono::high_resolution_clock::now();
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<std::string>("nn_graph_path", nn_graph_path);

  std::cout << "Building index" << std::endl;

  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "indexing time: " << diff.count() << "\n";
  index.Save(argv[6]);

  return 0;
}
