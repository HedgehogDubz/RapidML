#include "neural_network.hpp"

#include <vector>

class neural_network {
private:
    int inputs_;
    std::vector<int> hidden_layers_;
    int outputs_;
    std::vector<std::vector<double>> weights_;
    std::vector<std::vector<double>> biases_;

public:
    neural_network(int inputs, std::vector<int> hidden_layers, int outputs)
        : inputs_{inputs}, hidden_layers_{hidden_layers}, outputs_{outputs} {}
    ~neural_network() {}

    std::vector<int> feed_forward(std::vector<double> input) {}
};
