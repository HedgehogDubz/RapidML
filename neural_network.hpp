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
    std::vector<double> neurons_;

    std::vector<size_t> weight_layer_ends_;
    std::vector<size_t> neuron_layer_ends_;

    std::mt19937 generator_{std::random_device{}()};
    std::uniform_real_distribution<double> real_distribution_{-1.0, 1.0};


    double random() { return real_distribution_(generator_); }
    static constexpr double relu(double number) noexcept { return (number <= 0) ? 0 : number; }

public:
    neural_network(size_t inputs, std::vector<size_t> hidden_layers,
                   size_t outputs, init_type init_weights_type,
                   init_type init_biases_type);

    std::vector<double> run(std::vector<double> input);
};
