require "json"
require "./../learning_style.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        record NetConfig,
          hidden_layer_qty : Int32 = 2,
          hist_qty_max : Int32 = 2,
          time_col_qty : Int32 = 3,

          input_state_size : Int32 = 3,
          hidden_state_size : Int32 = 4,
          output_state_size : Int32 = 2,

          initial_bias_disabled : Bool = true,
          initial_bias_scale : Float64 = rand,

          # TODO: research what is the ideal default learning styles per layer
          hidden_learing_styles_first : LearningStyle = LS_TANH,
          hidden_learing_styles_middle : LearningStyle = LS_RELU,
          output_learing_style : LearningStyle = LS_SIGMOID,
          learning_rate : Float64? = nil,
          momentum : Float64? = nil,
          deriv_scale : Float64 = rand / 100.0,
          error_distance_history_max : Int32 = 10 do
            
          include JSON::Serializable
        end
      end
    end
  end
end

# icr -r ./src/ai4cr.cr

# rnn = Ai4cr::NeuralNetwork::Cmn::Rnn::Net.new
