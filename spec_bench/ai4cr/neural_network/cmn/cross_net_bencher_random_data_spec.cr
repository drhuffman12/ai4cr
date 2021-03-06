# Yeah, technically not a spec, but let's roll with this for now ...
require "../../../spec_bench_helper"
require "benchmark"
require "ascii_bar_charter"

# TODO: 'RandomData' is good too, but generalize this so we can take
# specific inputs, outputs, and hidden size/qty and run/compare each
# learning style (possibly w/ some other learning style params)

module Ai4cr
  class CrossNetBencherRandomData
    getter height : Int32, width : Int32
    getter learning_session_count : Int32, learning_session_indexes : Array(Int32)
    getter structure
    getter training_io_qty : Int32
    getter graph_sample_percent : Int32 # : Float64
    getter training_io_indexes : Array(Int32)
    getter height_indexes : Array(Int32)
    getter width_indexes : Array(Int32)
    getter example_input_set : Array(Array(Float64))
    getter example_output_set : Array(Array(Float64))
    getter example_input_set_tanh : Array(Array(Float64))
    getter example_output_set_tanh : Array(Array(Float64))

    getter net_backprop
    getter net_ls_prelu
    getter net_ls_relu
    getter net_ls_sigmoid
    getter net_ls_tanh

    def initialize(@height, @width, @learning_session_count = 5)
      @structure = [@height, @width]
      @learning_session_indexes = Array.new(learning_session_count) { |i| i }

      @training_io_qty = MULTI_TYPE_TEST_QTY * 10
      @graph_sample_percent = training_io_qty // QTY_X_PERCENT_DENOMINATOR # 20
      @training_io_indexes = training_io_qty.times.to_a

      @height_indexes = height.times.to_a
      @width_indexes = width.times.to_a

      @example_input_set = training_io_indexes.map { height_indexes.map { rand().to_f } }
      @example_output_set = training_io_indexes.map { width_indexes.map { rand().round.to_f } }

      @example_input_set_tanh = example_input_set.map { |a| a.map { |b| (b*2 - 1).to_f.round(2) } }
      @example_output_set_tanh = example_output_set.map { |a| a.map { |b| (b*2 - 1).to_f.round(2) } }

      @net_backprop = Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure)
      @net_ls_prelu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::LS_PRELU)
      @net_ls_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::LS_RELU)
      @net_ls_sigmoid = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::LS_SIGMOID)
      @net_ls_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::LS_TANH)
    end

    def run
      puts "\n==== compare_initialization ====\n"
      compare_initialization
      puts "\n==== compare_training ====\n"
      compare_training
      puts "\n==== compare_learning ====\n"
      # TODO: refactor Chain based on code cleanup doen for RnnSimple; then, refactor this test file (w/ Spectator)
      # compare_learning
    end

    def compare_initialization
      Benchmark.ips do |x|
        x.report("Initializing Backpropagation") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
        x.report("Initializing MiniNet (PRELU)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::LS_PRELU) }
        x.report("Initializing MiniNet (RELU)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::LS_RELU) }
        x.report("Initializing MiniNet (SIGMOID)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::LS_SIGMOID) }
        x.report("Initializing MiniNet (TANH)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::LS_TANH) }
      end
    end

    def compare_training
      Benchmark.ips do |x|
        x.report("Training (on random data) Backpropagation") do
          calc_training(net_backprop)
        end
        x.report("Training (on random data) MiniNet (PRELU)") do
          calc_training(net_ls_prelu)
        end
        x.report("Training (on random data) MiniNet (RELU)") do
          calc_training(net_ls_relu)
        end
        x.report("Training (on random data) MiniNet (SIGMOID)") do
          calc_training(net_ls_sigmoid)
        end
        x.report("Training (on random data) MiniNet (TANH)") do
          calc_training(net_ls_tanh, inputs: example_input_set_tanh, outputs: example_output_set_tanh)
        end
      end
    end

    def compare_learning
      calc_learning(net_backprop, net_name: "net_backprop")
      calc_learning(net_ls_prelu)
      calc_learning(net_ls_relu)
      calc_learning(net_ls_sigmoid)
      calc_learning(net_ls_tanh, inputs: example_input_set_tanh, outputs: example_output_set_tanh)
    end

    private def calc_training(net, inputs = example_input_set, outputs = example_output_set)
      training_io_indexes.each do |i|
        net.train(inputs[i], outputs[i])
      end
    end

    private def calc_learning(net, net_name = nil, inputs = example_input_set, outputs = example_output_set)
      learning_session_indexes.each do |lsi|
        net.init_network
        training_io_indexes.each do |i|
          net.train(example_input_set[i], example_output_set[i])
        end
        name = net_name.nil? ? net.learning_style : net_name
        plot_errors(name, net)
      end
    end
  end
end

cnb = Ai4cr::CrossNetBencherRandomData.new(100, 100)
cnb.run
