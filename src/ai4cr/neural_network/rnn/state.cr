require "json"
require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      struct State
        include JSON::Serializable
        include Ai4cr::NeuralNetwork::Backpropagation::Math

        alias ChannelPosition = NamedTuple(time_col_index: Int32, channel_set_index: Int32, channel_type: Symbol)

        property config : Config
        # property channel_set : Array(ChannelSet)

        # property channel_sets : Array(Array(Hash(Symbol, Array(NeuralNetwork::Backpropagation::Net))))
        property channel_sets : Hash(ChannelPosition, NeuralNetwork::Backpropagation::Net)
        # property channel_set_io_mappings : # time_col_index => channel_set_index => channel_symbol => 

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
          qty_lpfc_layers = 3,
          qty_time_cols_neighbor_inputs = 2,
          qty_recent_memory = 2,

          # Embedded Backpropagation Nets:
          @hidden_scale_factor = 2,
          structure_hidden_laters : Array(Int32)? = nil, disable_bias = true, learning_rate = nil, momentum = nil
        )
          @config = Config.new(
            qty_states_in, qty_states_out,
            qty_time_cols,
            qty_lpfc_layers, qty_time_cols_neighbor_inputs, qty_recent_memory,
            hidden_scale_factor, structure_hidden_laters, disable_bias, learning_rate, momentum
          )

          # @inputs = init_inputs
          # @outputs = init_outputs
          # @channel_set = init_channel_sets # (config) # (disable_bias)
          @channel_sets = init_channel_sets # (config) # (disable_bias)

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

        # def init_channel_sets # (disable_bias)
        #   (0..(config.qty_lpfc_layers - 1)).to_a.map do |channel_set_index|
        #     _disable_bias = channel_set_index > 0 # TODO: utilize disable_bias .. but only if ????
        #     ChannelSet.new(channel_set_index: channel_set_index, disable_bias: _disable_bias, config: config)
        #   end
        # end

        def init_channel_sets # (disable_bias)
          # alias ChannelPosition = NamedTuple(time_col_index: Int32, channel_set_index: Int32, channel_type: Symbol)
          # property channel_sets : Hash(ChannelPosition, NeuralNetwork::Backpropagation::Net)

          qty_states_in_and_out = config.qty_states_out + config.qty_states_out

          _learning_rate = config.learning_rate
          _momentum = config.momentum
          
          channel_set_index_max = config.qty_lpfc_layers - 1
          _channel_sets = Hash(ChannelPosition, NeuralNetwork::Backpropagation::Net).new
          (0..(config.qty_time_cols - 1)).each do |time_col_index|
            (0..channel_set_index_max).each do |channel_set_index|
              [:local, :past, :future, :combo].each do |channel_type|
                if channel_set_index == 0
                  _structure = [config.qty_states_in] + config.structure_hidden_laters + [qty_states_in_and_out]
                  _disable_bias = false # TODO: utilize disable_bias .. but only if ????
                elsif channel_set_index == channel_set_index_max
                  _structure = [qty_states_in_and_out] + config.structure_hidden_laters + [config.qty_states_out]
                  _disable_bias = true # TODO: utilize disable_bias .. but only if ????
                else
                  _structure = [qty_states_in_and_out] + config.structure_hidden_laters + [qty_states_in_and_out]
                  _disable_bias = true # TODO: utilize disable_bias .. but only if ????
                end
                
                _disable_bias = channel_set_index > 0 # TODO: utilize disable_bias .. but only if ????
                channel_position = {
                  time_col_index: time_col_index,
                  channel_set_index: channel_set_index,
                  channel_type: channel_type
                }
                _channel_sets[channel_position] = NeuralNetwork::Backpropagation::Net.new(
                  structure: _structure,
                  disable_bias: _disable_bias,
                  learning_rate: _learning_rate,
                  momentum: _momentum
                )
              end
            end
          end
          _channel_sets
        end
      end
    end
  end
end

# touch spec/ai4cr/neural_network/rnn/config_spec.cr