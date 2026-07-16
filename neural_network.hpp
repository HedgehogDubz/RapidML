#pragma once

#include <cstddef>
#include <random>
#include <vector>

enum class init_type
{
    zeroes,
    He
};

class neural_network
{
private:
    size_t inputs_;
    std::vector<size_t> hidden_layers_;
    size_t outputs_;

    std::vector<double> weights_;
    std::vector<double> biases_;

    std::vector<size_t> weight_layer_ends_;
    std::vector<size_t> bias_layer_ends_;

    std::mt19937 generator_{std::random_device{}()};
    std::uniform_real_distribution<double> real_distribution_{-1.0, 1.0};

    double random() { return real_distribution_(generator_); }
    double ReLU(double number) { return (number <= 0) ? 0 : number; }

public:
    neural_network(size_t inputs, std::vector<size_t> hidden_layers,
                   size_t outputs, init_type init_random_weights,
                   init_type init_random_biases);

    std::vector<double> feed_forward(std::vector<double> input) const;
};
