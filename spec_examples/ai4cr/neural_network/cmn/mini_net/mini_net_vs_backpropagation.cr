require "./../../../../spec_helper"
require "benchmark"

width = 100
height = 100
structure = [width, height]

results = Benchmark.ips do |x|
  x.report("Backpropagation") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("MiniNetExp") { Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(width: width, height: height) }
end

height_considering_bias = height + 1

results = Benchmark.ips do |x|
  x.report("Array.new") { Array.new(height_considering_bias) { |i| i } }
  x.report("to_a") { (0..(height_considering_bias-1)).to_a  }
end

def wip
  ################################################################################
  ################################################################################
  ################################################################################
  ################################################################################

  require "./src/ai4cr/neural_network/mini_net_exp.cr"
  require "./src/ai4cr/neural_network/mini_net_tanh.cr"
  require "./src/ai4cr/neural_network/mini_net_relu.cr"
  
  inputs_given = [0.1,0.2,0.3]
  outputs_expected = [1.0,0.0]

  # mn = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(width: outputs_expected.size, height: inputs_given.size)
  # mn = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(width: outputs_expected.size, height: inputs_given.size)
  mn = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(width: outputs_expected.size, height: inputs_given.size)
  mn.weights = [
    [-0.5,-1.0],
    [-0.5,-1.0],
    [-0.5,-1.0],
    [-0.5,-1.0],
  ]
  mn.learning_rate = 0.25
  mn.momentum = 0.1
  # puts mn.pretty_inspect
  result = mn.eval(inputs_given)
  #=> [0.31002551887238755, 0.16798161486607552]

  qty_training_sessions = 10
  error_list = Array(Float64).new(qty_training_sessions, 0.0)

  puts "\n"
  qty_training_sessions.times.each do |i|
    print "."
    error_list[i] = mn.train(inputs_given, outputs_expected)
  end
  puts "\n"

  puts mn.pretty_inspect
  puts error_list.pretty_inspect
  

  ################################################################################
  ################################################################################
  ################################################################################
  ################################################################################

  # require "./src/ai4cr/neural_network/mini_net_common.cr"
  
  require "./src/ai4cr/neural_network/mini_net_exp.cr"
  
  inputs_given = [0.1,0.2,0.3]
  outputs_expected = [1.0,0.0]

  # mn = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(width: inputs_given.size, height: outputs_expected.size)
  mn = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(width: outputs_expected.size, height: inputs_given.size)
  puts mn.pretty_inspect
  result = mn.eval(inputs_given)
  # => [0.02931223075135632, 0.0009110511944006454]

  qty_training_sessions = 500
  error_list = Array(Float64).new(qty_training_sessions, 0.0)

  puts "\n"
  qty_training_sessions.times.each do |i|
    print "."
    error_list[i] = mn.train(inputs_given, outputs_expected)
  end
  puts "\n"

  puts mn.pretty_inspect
  puts error_list.pretty_inspect

  puts mn.guesses_rounded
  
  puts mn.guesses_sorted
  puts mn.guesses_top_n
  puts mn.guesses_bottom_n


  ####

  require "./src/ai4cr/neural_network/mini_net_exp.cr"
  
  inputs_given = [0.1,0.2,0.3,0.4,0.5]
  outputs_expected = [1.0,0.0,1.0,1.0,0.0]

  # mn = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(width: inputs_given.size, height: outputs_expected.size)
  mn = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(width: outputs_expected.size, height: inputs_given.size)
  puts mn.pretty_inspect
  result = mn.eval(inputs_given)

  qty_training_sessions = 500
  error_list = Array(Float64).new(qty_training_sessions, 0.0)

  puts "\n"
  qty_training_sessions.times.each do |i|
    print "."
    error_list[i] = mn.train(inputs_given, outputs_expected)
  end
  puts "\n"

  puts mn.pretty_inspect
  puts error_list.pretty_inspect

  puts mn.guesses_rounded
  
  puts mn.guesses_sorted
  puts mn.guesses_top_n
  puts mn.guesses_bottom_n


  ####

  require "./src/ai4cr/neural_network/mini_net_tanh.cr"
  
  inputs_given = [-0.1,0.2,-0.3,0.4,-0.5]
  outputs_expected = [1.0,0.0,-1.0,1.0,0.0]

  # mn = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(width: inputs_given.size, height: outputs_expected.size)
  mn = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(width: outputs_expected.size, height: inputs_given.size)
  # puts mn.pretty_inspect
  result = mn.eval(inputs_given)

  qty_training_sessions = 100
  error_list = Array(Float64).new(qty_training_sessions, 0.0)

  puts "\n"
  qty_training_sessions.times.each do |i|
    print "."
    error_list[i] = mn.train(inputs_given, outputs_expected)
  end
  puts "\n"

  puts mn.pretty_inspect
  puts error_list.pretty_inspect

  puts mn.guesses_rounded
  
  puts mn.guesses_sorted
  puts mn.guesses_top_n
  puts mn.guesses_bottom_n

  puts mn.outputs_guessed


  ####

  require "./src/ai4cr/neural_network/mini_net_relu.cr"
  
  # inputs_given = [-0.1,0.2,-0.3,0.4,-0.5]
  # outputs_expected = [1.0,0.0,-1.0,1.0,0.0]


  inputs_given = [0.1,0.2,0.3,0.4,0.5]
  outputs_expected = [1.0,0.0,1.0,1.0,0.0]

  mn = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(width: inputs_given.size, height: outputs_expected.size)
  puts mn.pretty_inspect
  result = mn.eval(inputs_given)

  qty_training_sessions = 5
  error_list = Array(Float64).new(qty_training_sessions, 0.0)

  puts "\n"
  qty_training_sessions.times.each do |i|
    print "."
    error_list[i] = mn.train(inputs_given, outputs_expected)
  end
  puts "\n"

  puts mn.pretty_inspect
  puts error_list.pretty_inspect

  puts mn.guesses_rounded

  puts mn.guesses_ceiled  
  
  puts mn.guesses_sorted
  puts mn.guesses_top_n
  puts mn.guesses_bottom_n

  puts mn.outputs_guessed

end

