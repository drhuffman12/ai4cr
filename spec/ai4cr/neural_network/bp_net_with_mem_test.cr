require "./../../spec_helper"

describe Ai4cr::NeuralNetwork::BpNetWithMem do
  it "initialize" do
    structure = [4,8,2]

    bp_net_with_mem = Ai4cr::NeuralNetwork::BpNetWithMem.new(structure: structure, memory_size: 3) # , disable_bias, learning_rate, momentum, memory_size

    puts "*"*8
    puts "bp_net_with_mem: #{bp_net_with_mem.pretty_inspect}"

    expected_memory_set = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    bp_net_with_mem.memory_set.should eq(expected_memory_set)

    # puts "*"*8
    # bp_net_with_mem.store

    # puts "bp_net_with_mem: #{bp_net_with_mem.pretty_inspect}"

    # puts "range_memory_current_to_end_boundary: #{bp_net_with_mem.range_memory_current_to_end_boundary.pretty_inspect}"
    # puts "range_start_boundary_to_old_memory_current: #{bp_net_with_mem.range_start_boundary_to_old_memory_current.pretty_inspect}"


    # puts "*"*8
    # bp_net_with_mem.store

    # puts "bp_net_with_mem: #{bp_net_with_mem.pretty_inspect}"

    # puts "range_memory_current_to_end_boundary: #{bp_net_with_mem.range_memory_current_to_end_boundary.pretty_inspect}"
    # puts "range_start_boundary_to_old_memory_current: #{bp_net_with_mem.range_start_boundary_to_old_memory_current.pretty_inspect}"

  end

  describe "store" do
    it "after bp_net_with_mem was just initialized" do
      structure = [4,8,2]

      bp_net_with_mem = Ai4cr::NeuralNetwork::BpNetWithMem.new(structure: structure, memory_size: 3) # , disable_bias, learning_rate, momentum, memory_size

      expected_memory_set = [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]

      bp_net_with_mem.store
      bp_net_with_mem.memory_set.should eq(expected_memory_set)
      
      puts "*"*8
      puts "bp_net_with_mem: #{bp_net_with_mem.pretty_inspect}"
    end

    # it "after bp_net_with_mem was just initialized, initial 'store' done, and trained w/ some data" do
    it "after bp_net_with_mem was just initialized and trained w/ some data" do
      structure = [4,8,2]

      bp_net_with_mem = Ai4cr::NeuralNetwork::BpNetWithMem.new(structure: structure, memory_size: 3) # , disable_bias, learning_rate, momentum, memory_size

      # bp_net_with_mem.store

      ins = structure.first.times.map{rand}.to_a
      outs = structure.last.times.map{rand}.to_a
      bp_net_with_mem.net.train(ins,outs)


      expected_memory_sub_set_un_touched = [0.0, 0.0]
      # expected_memory_set = [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]

      bp_net_with_mem.store
      bp_net_with_mem.memory_set[0].should_not eq(expected_memory_sub_set_un_touched)
      bp_net_with_mem.memory_set[1].should eq(expected_memory_sub_set_un_touched)
      bp_net_with_mem.memory_set[2].should eq(expected_memory_sub_set_un_touched)
      
      puts "*"*8
      puts "bp_net_with_mem: #{bp_net_with_mem.pretty_inspect}"
    end

  end

end
