module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module BiDi
          # 'Pai' aka short for 'PropsAndInits'

          alias NodeInputSizes = Array(Array(NamedTuple(
            channel_forward: NamedTuple(
              # enabled: Bool,
              current_self_mem: Int32,
              sl_previous_input_or_combo: Int32,
              sl_previous_channel_forward: Int32,
              tc_previous_channel_forward: Int32),
            channel_backward: NamedTuple(
              # enabled: Bool,
              current_self_mem: Int32,
              sl_previous_input_or_combo: Int32,
              sl_previous_channel_backward: Int32,
              tc_next_channel_backward: Int32),
            channel_sl_or_combo: NamedTuple(
              current_self_mem: Int32,
              sl_previous_input_or_combo: Int32,
              current_forward: Int32, current_backward: Int32))))
          # alias MiniNetSet = Array(Array(Hash(Symbol, Hash(Symbol, Int32))))
          # alias MiniNetSet = Array(Array(Ai4cr::NeuralNetwork::Cmn::MiniNet)
          alias MiniNetSet = Array(Array(Hash(Symbol, Cmn::MiniNet)))

          alias Weights = Array(Array(Hash(Symbol, Array(Array(Float64)))))
        end
      end
    end
  end
end
