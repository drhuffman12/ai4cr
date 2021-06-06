module Ai4cr
  module NeuralNetwork
    module Pmn
      module Aliases
        alias Offset = Array(Int32)
        alias NodeCoord = NamedTuple(from_channel: String, from_offset: Offset)
        alias HeightSet = Hash(NodeCoord, Int32)
        alias HeightSetIndexes = Hash(NodeCoord, Array(Int32))
      end
    end
  end
end
