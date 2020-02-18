require "json"
require "./cmn/*"

module Ai4cr
  module NeuralNetwork
    module Cmn
      # Cmn aka Connectable Mini Networks
    end
  end
end

# # TODO:
# - Coordinate system
# - List of NodeCoordinates
#   - coordinate

# - List of NodeOutputs
#   - coordinate
#   - output size
#   - input_coordinates (from which we derive input sizeS)

# - List of NodeInputs
#   - node_inputs = node_outputs.each |node|

#     - input_nodes[node.input_coordinate] = my_node_set.node_at(node.input_coordinate)
#     - input_sizes[node.input_coordinate] = input_nodes[node.input_coordinate].output_size

# - List of Nodes
#   - output size
#   - equation type (e.g.: 'exp', 'relu', 'tanh')
# -
# - List of Node Coordinates
# - Chain of nodes
#   - parallel or serial