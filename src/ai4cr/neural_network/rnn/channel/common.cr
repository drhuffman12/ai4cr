# require "./math"
# require "./aliases"
require "./../node/*"
require "./interface"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Channel
        abstract class Common(T) # Ai4cr::NeuralNetwork::Rnn::Channel::Common
          include Channel::Interface

          getter time_column_qty : Int32
          getter time_column_range : Range(Int32, Int32)
          # property nodes : Array(Node::Common)
          # property nodes : Array(Node::Interface)
          getter dendrite_offsets : Array(Int32)
          getter state_qty : Int32
          
          property nodes : Array(T)
  
          def initialize(@time_column_qty = TIME_COLUMN_QTY_DEFAULT, @dendrite_offsets = DENDRITE_OFFSETS_DEFAULT, @state_qty = Node::Interface::STATE_QTY_DEFAULT)
            @time_column_range = (0..time_column_qty-1)
            @nodes = init_nodes(state_qty)
          end

          # abstract def node_class
          def node_class
            T
          end

          # abstract def init_nodes
          # # @nodes = time_column_range.map{|t| node_class.new }

          def init_nodes(state_qty)
            @nodes = time_column_range.map{|t| node_class.new(state_qty) }
          end
        end
      end
    end
  end
end
      