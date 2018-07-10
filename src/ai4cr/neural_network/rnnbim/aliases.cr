module Ai4cr
  module NeuralNetwork
    module Rnnbim # RNN, Bidirectional, Inversable Memory
      # To help clarify net data:
      alias ChannelKey = Symbol # TODO: change to Enum?
      alias FromChannelKey = Symbol # TODO: change to Enum?
      alias ToChannelKey = Symbol # TODO: change to Enum?
      alias LayerName = String # TODO: change to Enum?

      alias NodesSimple = Array(Float64)
      alias NodesChrono = Array(NodesSimple)
      alias NodesChannel = Hash(ChannelKey, NodesChrono)
      alias NodesLayer = Hash(ChannelKey, NodesChannel)
      alias NodesHidden = Array(NodesLayer)

      alias WeightsSimple = Array(Array(Float64))
      alias WeightsFromChannel = Hash(FromChannelKey, WeightsSimple | Int32)
      alias WeightsToChannel = Hash(ToChannelKey,Array(WeightsFromChannel))
      alias WeightsNetwork = Hash(LayerName,WeightsToChannel)

      # To help clarify weight meta data:
      alias ChronoSize = Int32
      alias LayerSize = Int32
      alias MetaChronoKey = Symbol
      alias MetaWeightsSimple = Hash(Symbol, ChronoSize)
      alias MetaWeightsFromChannel = Hash(FromChannelKey, MetaWeightsSimple)
      alias MetaWeightsAtTime = Hash(MetaChronoKey, MetaWeightsFromChannel | ChronoSize)
      alias MetaWeightsToChannel = Hash(ToChannelKey,MetaWeightsAtTime)
      alias MetaWeightsNetwork = Hash(LayerName,MetaWeightsToChannel)
    end
  end
end
