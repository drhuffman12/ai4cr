require "benchmark"
require "../../../spec_bench_helper"

height_considering_bias = MULTI_TYPE_TEST_QTY + 1
width = MULTI_TYPE_TEST_QTY
height_indexes = Array.new(MULTI_TYPE_TEST_QTY) { |i| i }
width_indexes = Array.new(MULTI_TYPE_TEST_QTY) { |i| i }

puts "\n==== compare Array.new vs map vs to_a ====\n"
Benchmark.ips do |x|
  x.report("Array.new") { Array.new(height_considering_bias) { |i| i } }
  x.report("array.map") { height_indexes.map { |i| i } }
  x.report("to_a") { (0..(height_considering_bias - 1)).to_a }
end

puts "\n==== compare misc reversing of an array ====\n"
Benchmark.ips do |x|
  x.report("Array.new max minus") { Array.new(height_considering_bias) { |i| height_considering_bias - i - 1 } }
  x.report("Array.new reversed") { Array.new(height_considering_bias) { |i| i }.reverse }
  x.report("to_a reversed") { (0..(height_considering_bias - 1)).to_a.reverse }
  x.report("to_a downto") { (height_considering_bias - 1).downto(0).to_a.reverse }
end

puts "\n==== compare nested array of random values (-1 .. 1) ====\n"
Benchmark.ips do |x|
  # pre-calc'd indexes x pre-calc'd indexes x rand
  x.report("pre-calc'd indexes x pre-calc'd indexes x rand") {
    weights = height_indexes.map { width_indexes.map { rand*2 - 1 } }
  }
  # Array.new x Array.new x rand
  x.report("Array.new x Array.new x rand") {
    weights = Array.new(height_considering_bias) {
      Array.new(width) { (rand*2 - 1) }
    }
  }
end
