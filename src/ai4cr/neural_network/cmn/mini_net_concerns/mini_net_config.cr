require "json"
require "./../learning_style.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        record MiniNetConfig,
          output_state_size : Int32 = 2,

          input_prev_layer_size : Int32 = -1,

          # i.e.: use outputs of previous N nodes in layer starting w/ closest previous later
          input_hist_set_sizes : Array(Int32) = [-1],

          bias_disabled : Bool = false,
          bias_scale : Float64 = rand,

          learning_style : LearningStyle = LS_RELU,
          learning_rate : Float64? = nil,
          momentum : Float64? = nil,
          deriv_scale : Float64 = rand / 100.0,
          error_distance_history_max : Int32 = 10 do
          include JSON::Serializable

          def height
            input_prev_layer_size + input_hist_set_sizes.sum
          end

          def width
            output_state_size
          end
        end
      end
    end
  end
end

# icr -r ./src/ai4cr.cr

# rnn = Ai4cr::NeuralNetwork::Cmn::Rnn::Net.new
