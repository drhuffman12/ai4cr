require "./../../spec_helper"

describe Ai4cr::NeuralNetwork::BpNetWithMem do
  it "initialize" do
    structure = [4,8,2]

    bp_net_with_mem = Ai4cr::NeuralNetwork::BpNetWithMem.new(structure: structure, memory_size: 3) # , disable_bias, learning_rate, momentum, memory_size

    bp_net_with_mem.store

    puts "bp_net_with_mem: #{bp_net_with_mem.pretty_inspect}"

    puts "range_memory_current_to_end_boundary: #{bp_net_with_mem.range_memory_current_to_end_boundary.pretty_inspect}"
    puts "range_start_boundary_to_old_memory_current: #{bp_net_with_mem.range_start_boundary_to_old_memory_current.pretty_inspect}"


    bp_net_with_mem.store

    puts "bp_net_with_mem: #{bp_net_with_mem.pretty_inspect}"

    puts "range_memory_current_to_end_boundary: #{bp_net_with_mem.range_memory_current_to_end_boundary.pretty_inspect}"
    puts "range_start_boundary_to_old_memory_current: #{bp_net_with_mem.range_start_boundary_to_old_memory_current.pretty_inspect}"

  end

end
