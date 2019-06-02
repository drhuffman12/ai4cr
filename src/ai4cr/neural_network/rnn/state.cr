require "json"
require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      struct State
        include JSON::Serializable
        include Ai4cr::NeuralNetwork::Backpropagation::Math

        property config : Config
        property channel_set : Array(ChannelSet)

        # include NeuralNetwork::Common::Initializers::LearningRate
        # include NeuralNetwork::Common::Initializers::Momentum

        # property qty_states_in
        # property qty_states_out
        # property qty_time_cols
        # property qty_lpfc_layers
        # property qty_recent_memory

        # property structure_hidden_laters : Array(Int32)
        # property disable_bias : Bool

        # property inputs  : Array(Array(Float64))
        # property outputs : Array(Array(Float64))

        # property local_channels  : Array(Array(NeuralNetwork::Backpropagation::Net))
        # property past_channels   : Array(Array(NeuralNetwork::Backpropagation::Net))
        # property future_channels : Array(Array(NeuralNetwork::Backpropagation::Net))
        # property combo_channels  : Array(Array(NeuralNetwork::Backpropagation::Net))

        def initialize(
          # RNN Net:
          qty_states_in = 3, qty_states_out = 4,
          qty_time_cols = 5,
          qty_lpfc_layers = 2,
          qty_time_cols_neighbor_inputs = 2,
          qty_recent_memory = 2,

          # Embedded Backpropagation Nets:
          structure_hidden_laters = [] of Int32, disable_bias = true, learning_rate = nil, momentum = nil
        )
          @config = Config.new(
            qty_states_in, qty_states_out,
            qty_time_cols,
            qty_lpfc_layers, qty_time_cols_neighbor_inputs, qty_recent_memory,
            structure_hidden_laters, disable_bias, learning_rate, momentum
          )

          # @inputs = init_inputs
          # @outputs = init_outputs
          @channel_set = init_channel_sets # (config) # (disable_bias)

          # @io_connections = init_io_connections
        end

        # def init_io_connections

        # end

        # def init_inputs
        #   (0..(config.qty_time_cols - 1)).to_a.map do |time_col_index|
        #     (0..(config.qty_states_in - 1)).to_a.map do |input_index|
        #       0.0
        #     end
        #   end
        # end

        # def init_outputs
        #   (0..(config.qty_time_cols - 1)).to_a.map do |time_col_index|
        #     (0..(config.qty_states_out - 1)).to_a.map do |input_index|
        #       0.0
        #     end
        #   end
        # end

        def init_channel_sets # (disable_bias)
          (0..(config.qty_lpfc_layers - 1)).to_a.map do |channel_set_index|
            _disable_bias = channel_set_index > 0 # TODO: utilize disable_bias .. but only if ????
            ChannelSet.new(channel_set_index: channel_set_index, disable_bias: _disable_bias, config: config)
          end
        end
      end
    end
  end
end

# touch spec/ai4cr/neural_network/rnn/config_spec.cr