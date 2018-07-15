module Ai4cr
  module NeuralNetwork
    module Rnnbim # RNN, Bidirectional, Inversable Memory
      # Math functions
      class Math
        # Math
        def self.node_delta_scale(hidden_layer_index) # time_column_index
          2 ** hidden_layer_index
        end
        
        def self.node_scaled_border_past(hidden_layer_index)
          node_delta_scale(hidden_layer_index)
        end
        
        def self.node_scaled_border_future(time_column_range, hidden_layer_index)
          time_column_range.max - node_delta_scale(hidden_layer_index)
        end

        def self.rnd_pos_neg_one
          rand*2 - 1.0
        end
      end
    end
  end
end
