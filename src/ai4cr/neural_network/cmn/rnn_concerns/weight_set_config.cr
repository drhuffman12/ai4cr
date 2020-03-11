require "json"
require "./../learning_style.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        record WeightSetConfig,
          initial_bias_enabled : Bool = true,
          initial_bias_scale : Float64 = rand,
  
          output_state_size : Int32 = 2,
  
          input_prev_layer_size : Int32 = -1,
  
          # i.e.: use outputs of previous N nodes in layer starting w/ closest previous later
          input_hist_set_sizes : Array(Int32) = [-1],
  
          learing_style : LearningStyle = LS_RELU do
          include JSON::Serializable
        end
      end
    end
  end
end

# icr -r ./src/ai4cr.cr

# rnn = Ai4cr::NeuralNetwork::Cmn::Rnn::Net.new
