#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <iostream>
#include <tinytensor/optimizer.hpp>
#include <tinytensor/shape.hpp>
#include <tinytensor/tensor.hpp>

using namespace tinytensor;

uint32_t read_u32_be(std::ifstream &file) {
  // reads from big endian u32
  unsigned char bytes[4];

  file.read(reinterpret_cast<char *>(bytes), 4);

  return (static_cast<uint32_t>(bytes[0]) << 24) |
         (static_cast<uint32_t>(bytes[1]) << 16) |
         (static_cast<uint32_t>(bytes[2]) << 8) |
         (static_cast<uint32_t>(bytes[3]));
}

struct MNISTDataset {
  std::vector<float> images;
  std::vector<uint8_t> labels;

  size_t size;
  size_t rows;
  size_t cols;
};

MNISTDataset load_mnist(const std::string &images_path,
                        const std::string &labels_path) {
  std::ifstream images_file(images_path, std::ios::binary);
  std::ifstream labels_file(labels_path, std::ios::binary);

  if (!images_file || !labels_file) {
    throw std::runtime_error("Couldn't open mnist file");
  }

  uint32_t images_magic = read_u32_be(images_file);
  uint32_t num_images = read_u32_be(images_file);
  uint32_t rows = read_u32_be(images_file);
  uint32_t cols = read_u32_be(images_file);

  uint32_t labels_magic = read_u32_be(labels_file);
  uint32_t num_labels = read_u32_be(labels_file);

  if (images_magic != 2051) {
    throw std::runtime_error("Invalid MNIST image file");
  }

  if (labels_magic != 2049) {
    throw std::runtime_error("Invalid MNIST label file");
  }

  if (num_images != num_labels) {
    throw std::runtime_error("Image/label count mismatch");
  }

  MNISTDataset dataset;
  dataset.size = num_images;
  dataset.rows = rows;
  dataset.cols = cols;

  dataset.images.resize((num_images)*rows * cols);
  dataset.labels.resize(num_labels);

  for (size_t i = 0; i < dataset.images.size(); i++) {
    uint8_t pixel;
    images_file.read(reinterpret_cast<char *>(&pixel), 1);
    dataset.images[i] = static_cast<float>(pixel) / 255.0f;
  }

  labels_file.read(reinterpret_cast<char *>(dataset.labels.data()),
                   dataset.labels.size());

  return dataset;
}

struct Batch {
  Tensor images;
  Tensor labels;
};

class MNISTLoader {
public:
  MNISTLoader(const MNISTDataset &dataset, size_t batch_size)
      : dataset(dataset), batch_size(batch_size) {
    reset();
  }

  void reset() {
    indices.resize(dataset.size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    position = 0;
  }

  bool has_next() const { return position < dataset.size; }

  Batch next() {
    if (position >= dataset.size) {
      throw std::out_of_range("No more batches");
    }
    size_t current_batch_size = std::min(batch_size, dataset.size - position);
    size_t num_pixels = dataset.rows * dataset.cols;
    std::vector<float> image_data(current_batch_size * num_pixels);
    std::vector<float> label_data(current_batch_size * 10);
    for (size_t i = 0; i < current_batch_size; i++) {
      size_t idx = indices[position + i];
      label_data[i * 10 + dataset.labels[idx]] = 1.0f;
      for (size_t j = 0; j < num_pixels; j++) {
        image_data[i * num_pixels + j] = dataset.images[idx * num_pixels + j];
      }
    }
    position += current_batch_size;
    Batch batch{Tensor(std::move(image_data), {current_batch_size, num_pixels}),
                Tensor(std::move(label_data), {current_batch_size, 10})};
    return batch;
  }

private:
  const MNISTDataset &dataset;
  size_t batch_size;
  size_t position = 0;

  std::vector<size_t> indices;
  std::mt19937 rng{std::random_device{}()};
};

int main() {
  auto train = load_mnist("data/mnist/train-images-idx3-ubyte",
                          "data/mnist/train-labels-idx1-ubyte");
  auto train_loader = MNISTLoader(train, 64);
  Tensor W1 = Tensor::kaiming_normal({784, 256}, true);
  Tensor B1({256}, true);
  Tensor W2 = Tensor::kaiming_normal({256, 128}, true);
  Tensor B2({128}, true);
  Tensor W3 = Tensor::kaiming_normal({128, 10}, true);
  Tensor B3({10}, true);

  SGD optim(0.01f, {&W1, &B1, &W2, &B2, &W3, &B3});

  for (size_t iterations = 0; iterations < 2; iterations++) {
    train_loader.reset();
    size_t batch_num = 0;
    while (train_loader.has_next()) {
      optim.zero_grad();
      Batch batch = train_loader.next();
      Tensor output = (batch.images.matmul(W1) + B1).relu();
      output = (output.matmul(W2) + B2).relu();
      output = (output.matmul(W3) + B3);

      Tensor diff = (output - batch.labels);
      Tensor loss = (diff * diff).mean();
      loss.backward();
      optim.step();
      std::cout << "Batches proccessed so far: " << ++batch_num
                << " Loss: " << loss.item() << std::endl;
    }
  }
  return 0;
}