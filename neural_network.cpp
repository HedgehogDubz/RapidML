#include "neural_network.hpp"

#include <algorithm>
#include <ranges>
#include <utility>

neural_network::neural_network(size_t inputs, std::vector<size_t> hidden_layers,
                               size_t outputs, init_type init_weights_type,
                               init_type init_biases_type)
    : inputs_{inputs},
      hidden_layers_{std::move(hidden_layers)},
      outputs_{outputs}
{
    // Compute total weight/bias counts and the per-layer end offsets.
    size_t weights_size = 0;
    size_t biases_size = 0;
    weight_layer_ends_.reserve(hidden_layers_.size() + 1);
    bias_layer_ends_.reserve(hidden_layers_.size() + 1);

    for (size_t i{0}; i < hidden_layers_.size() + 1; ++i)
    {
        const size_t previous_size =
            (i == 0) ? inputs_ : hidden_layers_[i - 1];
        const size_t current_size =
            (i == hidden_layers_.size()) ? outputs_ : hidden_layers_[i];

        weights_size += previous_size * current_size;
        weight_layer_ends_.push_back(weights_size);
        biases_size += current_size;
        bias_layer_ends_.push_back(biases_size);
    }
    weights_.resize(weights_size);
    biases_.resize(biases_size);

    if (init_weights_type == init_type::He)
    {
        size_t cur_weights_start = 0;
        for (size_t cur_layer_end : weight_layer_ends_)
        {
            auto start = weights_.begin() + cur_weights_start;
            auto end = weights_.begin() + cur_layer_end;
            std::ranges::generate(
                start, end, [this, cur_layer_end, cur_weights_start]()
                { return random() * 2 / (cur_layer_end - cur_weights_start); });

            cur_weights_start = cur_layer_end;
        }
    }
    if (init_biases_type == init_type::He)
    {
        size_t cur_biases_start = 0;
        for (size_t cur_layer_end : bias_layer_ends_)
        {
            auto start = biases_.begin() + cur_biases_start;
            auto end = biases_.begin() + cur_layer_end;
            std::ranges::generate(
                start, end, [this, cur_layer_end, cur_biases_start]()
                { return random() * 2 / (cur_layer_end - cur_biases_start); });

            cur_biases_start = cur_layer_end;
        }
    }
}

std::vector<double> neural_network::feed_forward(std::vector<double> input) const
{
    // TODO: implement the forward pass.
    return {};
}
