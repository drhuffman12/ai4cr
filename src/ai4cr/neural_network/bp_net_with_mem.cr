module Ai4cr
  module NeuralNetwork
    class BpNetWithMem
      property memory_size : Int32
      property memory_current_start_index : Int32
      property memory_current_end_index : Int32
      property net : Backpropagation
      property output_size : Int32
      property memory_set : Array(Array(Float64))
      # property foo

      def initialize(structure : Array(Int32), disable_bias : Bool? = nil, learning_rate : Float64? = nil, momentum : Float64? = nil, @memory_size : Int32 = 1)
        @net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure, disable_bias, learning_rate, momentum)
        @output_size = structure.last
        @memory_current_start_index = 0
        @memory_current_end_index = memory_size - 1
        # @memory_set = StaticArray(StaticArray, @memory_size).new(StaticArray(Float64, @output_size).new(0.0))
        # @memory_set = Array(Array(Float64)).new(memory_size) # { output_size.times.map{0.0} }
        @memory_set = memory_size.times.map{output_size.times.map{0.0}.to_a}.to_a
      end

      def store
        load_memory
        cycle_memory_indexes
      end

      def recall
        
      end

      def range_memory_current_to_end_boundary
        memory_current_start_index..(memory_size - 1)
      end

      def range_start_boundary_to_old_memory_current
        0..memory_current_end_index
      end

      # private

      private def cycle_memory_indexes
        @memory_current_start_index = (memory_current_start_index + 1) % memory_size
        @memory_current_end_index = (memory_current_end_index + 1) % memory_size
      end

      private def load_memory
        net.activation_nodes.last.each_with_index {|node_value, node_index| @memory_set[memory_current_start_index][node_index] = node_value }
      end
    end
  end
end
