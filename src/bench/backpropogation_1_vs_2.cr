require "benchmark"

require "../ai4cr"
# require "spec/test_helper"
require "../../spec/spec_helper"
require "../../spec_examples/support/neural_network/data/training_patterns"
require "../../spec_examples/support/neural_network/data/patterns_with_noise"
require "../../spec_examples/support/neural_network/data/patterns_with_base_noise"

# USAGE:
# time crystal build --release src/bench/backpropogation_1_vs_2.cr -o bin/bench/backpropogation_1_vs_2
# real    0m8.305s
# user    0m8.625s
# sys     0m0.167s
#
# time bin/bench/backpropogation_1_vs_2
# mkdir -p tmp/bench
# valgrind --tool=callgrind --inclusive=yes --tree=both --auto=yes --cache-sim=yes --branch-sim=yes --callgrind-out-file=tmp/bench/backpropogation_1_vs_2.callgrind.out bin/bench/backpropogation_1_vs_2
# valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes --callgrind-out-file=tmp/bench/backpropogation_1_vs_2.callgrind.out bin/bench/backpropogation_1_vs_2

def bench_check_guess(next_guess, expected_label)
  p(result_label(next_guess) == expected_label ? "." : "F")
end

correct_count = 0

error_averages = [] of Float64
is_a_triangle = [1.0, 0.0, 0.0]
is_a_square = [0.0, 1.0, 0.0]
is_a_cross = [0.0, 0.0, 1.0]

tr_input = TRIANGLE.flatten.map { |input| input.to_f / 5.0 }
sq_input = SQUARE.flatten.map { |input| input.to_f / 5.0 }
cr_input = CROSS.flatten.map { |input| input.to_f / 5.0 }

tr_with_noise = TRIANGLE_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }
sq_with_noise = SQUARE_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }
cr_with_noise = CROSS_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }

tr_with_base_noise = TRIANGLE_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }
sq_with_base_noise = SQUARE_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }
cr_with_base_noise = CROSS_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }


qty = 100 # (ARGV[0] || 10).to_i # 10 + (rand * 100).to_i
# shape = [256, 3]
shape = [256, 1000, 3]
# shape = [256, 500, 500, 3]
learning_rate = rand

puts "qty: #{qty}"
puts "shape: #{shape}"
puts "learning_rate: #{learning_rate}"

def train(net, qty, tr_input, sq_input, cr_input, is_a_triangle, is_a_square, is_a_cross)
  qty.times do # |i|
    # errors = {} of Symbol => Float64
    # [:tr, :sq, :cr].each do |s|
    #   case s
    #   when :tr
    #     # errors[:tr] = net.train(tr_input, is_a_triangle)
    #     net.train(tr_input, is_a_triangle)
    #   when :sq
    #     # errors[:sq] = net.train(sq_input, is_a_square)
    #     net.train(sq_input, is_a_square)
    #   when :cr
    #     # errors[:cr] = net.train(cr_input, is_a_cross)
    #     net.train(cr_input, is_a_cross)
    #   end
    # end

    [[tr_input, is_a_triangle], [sq_input, is_a_square], [cr_input, is_a_cross]].each do |io|
      net.train(io[0], io[1])  
    end
    # error_averages << (errors[:tr].to_f + errors[:sq].to_f + errors[:cr].to_f) / 3.0
  end

  next_guess = guess(net, tr_input)
  # bench_check_guess(next_guess, "TRIANGLE")

  next_guess = guess(net, sq_input)
  # bench_check_guess(next_guess, "SQUARE")

  next_guess = guess(net, cr_input)
  # bench_check_guess(next_guess, "CROSS")
end

Benchmark.ips do |x|
  x.report("Backpropagation") do
    net = Ai4cr::NeuralNetwork::Backpropagation.new(shape)
    net.learning_rate = learning_rate
    train(net, qty, tr_input, sq_input, cr_input, is_a_triangle, is_a_square, is_a_cross)
  end
  
  x.report("Backpropagation2") do
    net = Ai4cr::NeuralNetwork::Backpropagation2.new(shape)
    net.learning_rate = learning_rate
    train(net, qty, tr_input, sq_input, cr_input, is_a_triangle, is_a_square, is_a_cross)
  end
end
