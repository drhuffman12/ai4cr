module Ai4cr
  module NeuralNetwork
    module Common
      module Initializers
        module StructureHiddenLaters
          # DEFAULT_LEARNING_RATE = 0.25
  
          property structure_hidden_laters : Array(Int32)

          # # def init_structure_hidden_laters(_structure_hidden_laters, _qty_states_in, _qty_states_out, _hidden_scale_factor = 2)
          # def init_structure_hidden_laters(_qty_hidden_laters, _qty_states_in, _qty_states_out, _hidden_scale_factor = 2)
          #   # must be positive
          #   qty_states = (_hidden_scale_factor * (_qty_states_in + _qty_states_out)).round.to_i32
          #   # _structure_hidden_laters.nil? ? [qty_states] : _structure_hidden_laters.as(Array(Int32))
          #   _qty_hidden_laters.times.to_a.map{|layer| qty_states}
          # end  

          # def init_structure_hidden_laters(_qty_states_in, _qty_states_channel_out, _qty_states_out, _qty_hidden_laters)
          def init_structure_hidden_laters(_qty_states_channel_out, _qty_hidden_laters)
            _qty_hidden_laters.times.to_a.map { _qty_states_channel_out }
          end
        end
      end
    end
  end
end
