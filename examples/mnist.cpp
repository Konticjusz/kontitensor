#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm>
#include <span>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>


#include <tinytensor/tensor.hpp>
#include <tinytensor/shape.hpp>
#include <tinytensor/optimizer.hpp>


uint32_t read_u32_be(std::ifstream& file){
    // reads from big endian u32
    unsigned char bytes[4];

    file.read(reinterpret_cast<char*>(bytes), 4);

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

MNISTDataset load_mnist(const std::string& images_path, const std::string& labels_path){
    std::ifstream images_file(images_path, std::ios::binary);
    std::ifstream labels_file(labels_path, std::ios::binary);
    
    if (!images_file || !labels_file){
        throw std::runtime_error("Couldn't open mnist file");
    }

    uint32_t images_magic = read_u32_be(images_file);
    uint32_t num_images   = read_u32_be(images_file);
    uint32_t rows         = read_u32_be(images_file);
    uint32_t cols         = read_u32_be(images_file);

    uint32_t labels_magic = read_u32_be(labels_file);
    uint32_t num_labels   = read_u32_be(labels_file);

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

    dataset.images.resize((num_images) * rows * cols);
    dataset.labels.resize(num_labels);

    for (size_t i = 0; i < dataset.images.size(); i++){
        uint8_t pixel;
        images_file.read(reinterpret_cast<char*>(&pixel), 1);
        dataset.images[i] = static_cast<float>(pixel) / 255.0f;
    }

    labels_file.read(reinterpret_cast<char*>(dataset.labels.data(), dataset.labels.size()));
}

int main(){
    auto train = load_mnist(
        "data/mnist/train-images-idx3-ubyte",
        "data/mnist/train-labels-idx1-ubyte"
    );    
}